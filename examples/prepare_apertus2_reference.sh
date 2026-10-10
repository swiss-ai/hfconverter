#!/bin/bash
# Build an Apertus2-family reference from a config, custom code, and pinned tokenizer.
set -euo pipefail

OUTPUT="${1:?Set a new reference output directory under scratch}"
CONFIG_SOURCE="${2:-}"
EXAMPLES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "$CONFIG_SOURCE" ]]; then
    [[ -f "$CONFIG_SOURCE/config.json" ]] || { echo "Missing config.json in $CONFIG_SOURCE" >&2; exit 2; }
fi
COMMIT=28ad57a2757f6f72edb0def57ee725b6812f2df3
SOURCE="https://media.githubusercontent.com/media/swiss-ai/apertus-tokenizer-development/$COMMIT/preliminary_mul_200k"

[[ ! -e "$OUTPUT" ]] || {
    echo "Reference output already exists: $OUTPUT" >&2
    exit 2
}
mkdir -p "$OUTPUT"
if [[ -n "$CONFIG_SOURCE" ]]; then
    cp "$CONFIG_SOURCE/config.json" "$OUTPUT/"
else
    cp "$EXAMPLES/apertus2-chonk-hf-reference/config.json" "$OUTPUT/"
fi
cp "$EXAMPLES/apertus2-chonk-hf-reference/configuration_apertus2.py" "$OUTPUT/"

for name in tokenizer.json tokenizer_config.json special_tokens_map.json; do
    curl --fail --location --retry 3 \
        "$SOURCE/$name" \
        --output "$OUTPUT/$name"
done

(
    cd "$OUTPUT"
    sha256sum --check <<'SOURCE_HASHES'
cd403d3f219e2433e3f78b32644b8e6a6134668e15138e6546360330635a96b9  tokenizer.json
7e6b68d5a41fd06d399143b7591df15abf7ae2846fa8f444f66f3d5edee5996f  tokenizer_config.json
816ec96e37c6d15e3cbc535dc146c898a7218f209fc154384f31fc1e6ad31ba5  special_tokens_map.json
SOURCE_HASHES
    patch --batch --fuzz=0 -p1 < "$EXAMPLES/apertus2-chonk-hf-reference/remove-auto-eos.patch"
    # Match the exact tokenizer files used in the validated Apertus2 Chonk conversion.
    sha256sum --check <<'PATCHED_HASHES'
c85903769250eaa7553d83f13a2e2bf96866e76b320a581210d4bfddf820b020  tokenizer.json
3bcdd76f9ab09d6b89f914d108692275c8dad3b93e59396baf2016ff72ae2a15  tokenizer_config.json
816ec96e37c6d15e3cbc535dc146c898a7218f209fc154384f31fc1e6ad31ba5  special_tokens_map.json
PATCHED_HASHES
)
