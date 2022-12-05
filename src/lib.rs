use {
    clap::Parser,
    memmap::Mmap,
    std::{
        fs::File,
        io::{Error, ErrorKind, Result as IoResult},
        str::{from_utf8, Utf8Error},
    },
};

/// Arguments for program execution
///
/// Currently, this is just an input file path, but it may be more later. The default is
/// intentionally left empty such that multiple example programs can use the same struct without
/// needing to re-define it with a different default path.
#[derive(Parser)]
pub struct Args {
    /// Input file path
    #[arg(short, long, default_value_t)]
    input_file_path: String,
}

impl Args {
    /// Returns the input file path, or a provided default if the field is empty
    ///
    /// # Arguments
    ///
    /// * `default` - A default input file path string slice to use if `self.input_file_path` is
    ///   empty
    pub fn input_file_path<'a>(&'a self, default: &'a str) -> &'a str {
        if self.input_file_path.is_empty() {
            default
        } else {
            &self.input_file_path
        }
    }
}

/// Opens a memory-mapped UTF-8 file at a specified path, and passes in a `&str` over the file to a
/// provided callback function
///
/// # Arguments
///
/// * `file_path` - A string slice file path to open as a read-only file
/// * `f` - A callback function to invoke on the contents of the file as a string slice
///
/// # Errors
///
/// This function returns a `Result::Err`-wrapped `std::io::Error` if an error has occurred.
/// Possible causes are:
///
/// * `std::fs::File::open` was unable to open a read-only file at `file_path`
/// * `memmap::Mmap::map` fails to create an `Mmap` instance for the opened file
/// * `std::str::from_utf8` determines the file is not in valid UTF-8 format
///
/// `f` is only executed *iff* an error is not encountered.
///
/// # Safety
///
/// This function uses `Mmap::map`, which is an unsafe function. There is no guarantee that an
/// external process won't modify the file after it is opened as read-only.
///
/// # Undefined Behavior
///
/// Related to the **Safety** section above, it is UB if the opened file is modified by an external
/// process while this function is referring to it as an immutable string slice. For more info on
/// this, see:
///
/// * https://www.reddit.com/r/rust/comments/wyq3ih/why_are_memorymapped_files_unsafe/
/// * https://users.rust-lang.org/t/how-unsafe-is-mmap/19635
/// * https://users.rust-lang.org/t/is-there-no-safe-way-to-use-mmap-in-rust/70338
pub unsafe fn open_utf8_file<F: FnOnce(&str)>(file_path: &str, f: F) -> IoResult<()> {
    let file: File = File::open(file_path)?;

    // SAFETY: This operation is unsafe
    let mmap: Mmap = Mmap::map(&file)?;
    let bytes: &[u8] = &mmap;
    let utf8_str: &str = from_utf8(bytes).map_err(|utf8_error: Utf8Error| -> Error {
        Error::new(ErrorKind::InvalidData, utf8_error)
    })?;

    f(utf8_str);

    Ok(())
}
