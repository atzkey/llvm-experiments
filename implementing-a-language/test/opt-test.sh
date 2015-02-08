LLVM_PATH=/usr/local/Cellar/llvm/3.5.1/bin
$LLVM_PATH/llvm-as < opt-test.ir | $LLVM_PATH/opt -analyze -view-cfg
