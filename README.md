# Rust Nightly
Using stabilizer-lockin currently requires rust nightly since it uses [const generics](https://rust-lang.github.io/rfcs/2000-const-generics.html) to parameterize the ADC buffer size, number of external reference timestamps and downsampled output size. However, this use is not strictly necessary. Instead, we would use global constants and stable rust.
