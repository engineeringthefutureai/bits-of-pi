// tf_dump: teacher-forced forward pass that writes log-probabilities at every position.
//
// Built against the same llama.cpp commit and backend as llama-server, so E2 numbers
// come from the same arithmetic as the generation runs.
//
// usage: tf_dump -m model.gguf -f text.txt -o out_prefix [-ngl 999] [-b 2048] [-ub 512] [-no-mmap]
//
// writes:
//   <out_prefix>.tokens.json   {"n_vocab": V, "tokens": [...], "pieces": [...]}
//   <out_prefix>.logprobs.f32  float32 [n_tokens][V]; row i is log p(next token | tokens[0..i])

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string json_escape(const std::string & s) {
    std::string out;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string model_path, text_path, out_prefix;
    int ngl = 999, n_batch = 2048, n_ubatch = 512;
    bool mmap = true;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")   model_path = next();
        else if (a == "-f")   text_path  = next();
        else if (a == "-o")   out_prefix = next();
        else if (a == "-ngl") ngl        = std::stoi(next());
        else if (a == "-b")   n_batch    = std::stoi(next());
        else if (a == "-ub")  n_ubatch   = std::stoi(next());
        else if (a == "-no-mmap") mmap = false;
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || text_path.empty() || out_prefix.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -f text.txt -o out_prefix [-ngl N] [-b N] [-ub N] [-no-mmap]\n", argv[0]);
        return 1;
    }

    std::ifstream tf(text_path, std::ios::binary);
    std::stringstream ss;
    ss << tf.rdbuf();
    const std::string text = ss.str();

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    if (!mmap) mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // same tokenization as the server applies to a raw /completion prompt
    const int n_tok = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, false);
    std::vector<llama_token> tokens(n_tok);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), n_tok, true, false) < 0) {
        fprintf(stderr, "tokenization failed\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_tok + 16;
    cparams.n_batch         = n_batch;
    cparams.n_ubatch        = n_ubatch;
    cparams.n_seq_max       = 1;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k          = GGML_TYPE_F16;
    cparams.type_v          = GGML_TYPE_F16;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    std::ofstream lp_out(out_prefix + ".logprobs.f32", std::ios::binary);
    std::vector<float> row(n_vocab);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int start = 0; start < n_tok; start += n_batch) {
        const int n = std::min(n_batch, n_tok - start);
        batch.n_tokens = n;
        for (int j = 0; j < n; j++) {
            batch.token[j]     = tokens[start + j];
            batch.pos[j]       = start + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = 1;
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at %d\n", start); return 1; }

        for (int j = 0; j < n; j++) {
            const float * logits = llama_get_logits_ith(ctx, j);
            double mx = -INFINITY;
            for (int v = 0; v < n_vocab; v++) mx = std::max(mx, (double) logits[v]);
            double sum = 0.0;
            for (int v = 0; v < n_vocab; v++) sum += std::exp((double) logits[v] - mx);
            const double lse = mx + std::log(sum);
            for (int v = 0; v < n_vocab; v++) row[v] = (float) ((double) logits[v] - lse);
            lp_out.write((const char *) row.data(), sizeof(float) * n_vocab);
        }
    }
    llama_batch_free(batch);
    lp_out.close();

    std::ofstream tok_out(out_prefix + ".tokens.json");
    tok_out << "{\"n_vocab\": " << n_vocab << ", \"tokens\": [";
    for (int i = 0; i < n_tok; i++) tok_out << (i ? ", " : "") << tokens[i];
    tok_out << "], \"pieces\": [";
    for (int i = 0; i < n_tok; i++) {
        char buf[256];
        const int k = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), 0, true);
        tok_out << (i ? ", " : "") << "\"" << json_escape(std::string(buf, k > 0 ? k : 0)) << "\"";
    }
    tok_out << "]}\n";

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
