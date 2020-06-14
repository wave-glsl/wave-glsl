HTTP_PORT=2281;
TEST_URL="http://localhost:$HTTP_PORT/web";
OUT_DIR=./out;
WASM_DIR=./wasm;

echo "Build started";
mkdir -p "$OUT_DIR";

echo "Building wave.wasm";
emcc $WASM_DIR/*.cc --bind -O3 \
  -s WASM=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s FILESYSTEM=0 \
  -o "$OUT_DIR/wasm.js" || exit 1;

echo "Starting HTTP server on port $TEST_URL";
killall -q webfsd;
webfsd -F -l - -p $HTTP_PORT -r . -f index.html;
