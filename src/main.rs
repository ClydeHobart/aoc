use {aoc::*, clap::Parser};

fn main() {
    solutions().run(&<Args as Parser>::parse());
}
