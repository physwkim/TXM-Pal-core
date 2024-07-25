# TXM-Pal-core
### Build library
1) Install rustup
    https://rustup.rs/#
2) Download submodule
    - git submodule update --init --recursive
3) cd TXM-Pal/lib
4) conda activate txm-pal
5) Install library
    maturin develop -r

6) If the build fails, try running the following commands
       - cargo update -p maturin --precise 1.3.0
