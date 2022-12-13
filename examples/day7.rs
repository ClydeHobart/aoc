use file_descriptor::FileDescriptor;

use {
    aoc_2022::*,
    std::{
        fmt::{Debug, DebugStruct, Formatter, Result as FmtResult},
        iter::Peekable,
        mem::transmute,
        num::ParseIntError,
        ops::{Deref, Range},
        str::{FromStr, Split},
    },
};

enum ChangeDirectory<'s> {
    Root,
    Parent,
    Child(&'s str),
}

#[derive(Debug)]
pub enum ChangeDirectoryParseError<'s> {
    InvalidPattern(&'s str),
}

fn str_is_ascii_lowercase(s: &str) -> bool {
    s.chars().all(|c: char| c.is_ascii_lowercase())
}

impl<'s1, 's2: 's1> TryFrom<&'s2 str> for ChangeDirectory<'s1> {
    type Error = ChangeDirectoryParseError<'s1>;

    fn try_from(change_directory_str: &'s2 str) -> Result<Self, Self::Error> {
        match change_directory_str {
            "/" => Ok(Self::Root),
            ".." => Ok(Self::Parent),
            child_str if str_is_ascii_lowercase(child_str) => Ok(Self::Child(child_str)),
            invalid_pattern_str => Err(Self::Error::InvalidPattern(invalid_pattern_str)),
        }
    }
}

enum Command<'s> {
    ChangeDirectory(ChangeDirectory<'s>),
    List,
}

#[derive(Debug)]
pub enum CommandParseError<'s> {
    NoDollarSignToken,
    InvalidDollarSignToken(&'s str),
    NoCommandToken,
    InvalidCommandToken(&'s str),
    NoChangeDirectoryToken,
    FailedToParseChangeDirectory(ChangeDirectoryParseError<'s>),
}

impl<'s, 'i, 't: 's, T: Iterator<Item = &'t str>> TryFrom<TokenStream<'i, 't, T>> for Command<'s> {
    type Error = CommandParseError<'s>;

    fn try_from(mut command_token_iter: TokenStream<'i, 't, T>) -> Result<Self, Self::Error> {
        use CommandParseError as Error;

        match command_token_iter.next() {
            None => Err(Error::NoDollarSignToken),
            Some("$") => Ok(()),
            Some(invalid_dollar_sign_token) => {
                Err(Error::InvalidDollarSignToken(invalid_dollar_sign_token))
            }
        }?;

        match command_token_iter.next() {
            None => Err(Error::NoCommandToken),
            Some("cd") => match command_token_iter.next() {
                None => Err(Error::NoChangeDirectoryToken),
                Some(change_directory_str) => Ok(Self::ChangeDirectory(
                    change_directory_str
                        .try_into()
                        .map_err(Error::FailedToParseChangeDirectory)?,
                )),
            },
            Some("ls") => Ok(Self::List),
            Some(invalid_command_token) => Err(Error::InvalidCommandToken(invalid_command_token)),
        }
    }
}

mod file_descriptor {
    use super::*;

    const LEN_MASK: usize = usize::MAX >> 1_u32;
    const TYPE_MASK: usize = !LEN_MASK;

    #[derive(Debug)]
    pub enum FileParseError<'s> {
        NoSizeToken,
        FailedToParseSize(ParseIntError),
        NoNameToken,
        FileNameNotAsciiLowercase(&'s str),
        ExtensionNot3Characters(&'s str),
        ExtensionNotAsciiLowercase(&'s str),
    }

    #[derive(Debug)]
    pub enum DirectoryParseError<'s> {
        NoDirToken,
        InvalidDirToken(&'s str),
        NoNameToken,
    }

    pub struct FileDescriptor<'s> {
        pub name: &'s str,
        size: usize,
        start: usize,
        len: usize,
    }

    impl<'s> FileDescriptor<'s> {
        pub fn new(name: &'s str, is_directory: bool) -> Self {
            Self {
                name,
                size: usize::MAX,
                start: usize::MAX,
                len: if is_directory { usize::MAX } else { LEN_MASK },
            }
        }

        #[inline]
        pub fn set_size(&mut self, size: usize) {
            self.size = size;
        }

        #[inline]
        pub fn has_size(&self) -> bool {
            self.size != usize::MAX
        }

        pub fn get_size(&self) -> Option<usize> {
            if self.has_size() {
                Some(self.size)
            } else {
                None
            }
        }

        #[inline]
        pub fn is_directory(&self) -> bool {
            self.type_mask() == TYPE_MASK
        }

        pub fn get_directory<'f>(&'f self) -> Option<Directory<'f, 's>> {
            if self.is_directory() {
                Some(Directory(self))
            } else {
                None
            }
        }

        pub fn get_directory_mut<'f>(&'f mut self) -> Option<DirectoryMut<'f, 's>> {
            if self.is_directory() {
                Some(DirectoryMut(self))
            } else {
                None
            }
        }

        #[inline]
        fn type_mask(&self) -> usize {
            self.len & TYPE_MASK
        }

        fn try_file_from_token_stream<'i, T: Iterator<Item = &'s str>>(
            mut file_parse_iter: TokenStream<'i, 's, T>,
        ) -> Result<Self, FileParseError<'s>> {
            use FileParseError as Error;

            let size: usize = usize::from_str(file_parse_iter.next().ok_or(Error::NoSizeToken)?)
                .map_err(Error::FailedToParseSize)?;

            let name: &str = file_parse_iter.next().ok_or(Error::NoNameToken)?;
            let mut name_iter: Split<char> = name.split('.');

            match name_iter.next() {
                None => Err(Error::NoNameToken),
                Some(name) => {
                    if !str_is_ascii_lowercase(name) {
                        Err(Error::FileNameNotAsciiLowercase(name))
                    } else {
                        Ok(())
                    }
                }
            }?;

            match name_iter.next() {
                None => Ok(()),
                Some(ext) => {
                    if ext.len() != 3_usize {
                        Err(Error::ExtensionNot3Characters(ext))
                    } else if !str_is_ascii_lowercase(ext) {
                        Err(Error::ExtensionNotAsciiLowercase(ext))
                    } else {
                        Ok(())
                    }
                }
            }?;

            let mut file_descriptor: FileDescriptor<'s> = Self::new(name, false);

            file_descriptor.set_size(size);

            Ok(file_descriptor)
        }

        fn try_directory_from_token_stream<'i, T: Iterator<Item = &'s str>>(
            mut directory_parse_iter: TokenStream<'i, 's, T>,
        ) -> Result<Self, DirectoryParseError<'s>> {
            use DirectoryParseError as Error;

            match directory_parse_iter.next() {
                None => Err(Error::NoDirToken),
                Some("dir") => Ok(()),
                Some(invalid_dir_token) => Err(Error::InvalidDirToken(invalid_dir_token)),
            }?;

            match directory_parse_iter.next() {
                None => Err(Error::NoNameToken),
                Some(name) => Ok(Self::new(name, true)),
            }
        }
    }

    impl<'s> Debug for FileDescriptor<'s> {
        fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
            let mut debug_struct: DebugStruct = f.debug_struct("FileDescriptor");

            debug_struct
                .field("name", &self.name)
                .field("size", &self.size);

            if let Some(directory) = self.get_directory() {
                debug_struct.field("children", &directory.get_children());
            }

            debug_struct.finish()
        }
    }

    #[derive(Debug)]
    pub enum FileDescriptorParseError<'s> {
        NoInitialToken,
        FailedToParseDirectory(DirectoryParseError<'s>),
        FailedToParseFile(FileParseError<'s>),
    }

    impl<'s, 'i, T: Iterator<Item = &'s str>> TryFrom<TokenStream<'i, 's, T>> for FileDescriptor<'s> {
        type Error = FileDescriptorParseError<'s>;

        fn try_from(
            mut file_descriptor_parse_iter: TokenStream<'i, 's, T>,
        ) -> Result<Self, Self::Error> {
            use FileDescriptorParseError as Error;

            let mut peekable_output_parse_iter: Peekable<_> =
                file_descriptor_parse_iter.by_ref().peekable();

            match peekable_output_parse_iter.peek() {
                None => Err(Error::NoInitialToken),
                Some(&"dir") => Ok(Self::try_directory_from_token_stream(TokenStream::new(
                    &mut peekable_output_parse_iter,
                ))
                .map_err(Error::FailedToParseDirectory)?),
                Some(_) => Ok(Self::try_file_from_token_stream(TokenStream::new(
                    &mut peekable_output_parse_iter,
                ))
                .map_err(Error::FailedToParseFile)?),
            }
        }
    }

    pub struct Directory<'f, 's: 'f>(&'f FileDescriptor<'s>);

    impl<'f, 's: 'f> Directory<'f, 's> {
        #[inline]
        pub fn has_start(&self) -> bool {
            self.0.start != usize::MAX
        }

        pub fn get_start(&self) -> Option<usize> {
            if self.has_start() {
                Some(self.0.start)
            } else {
                None
            }
        }

        #[inline]
        pub fn has_end(&self) -> bool {
            (self.0.len & LEN_MASK) != LEN_MASK
        }

        pub fn get_end(&self) -> Option<usize> {
            if self.has_end() {
                Some(self.0.start + (self.0.len & LEN_MASK))
            } else {
                None
            }
        }

        #[inline]
        pub fn has_start_and_end(&self) -> bool {
            self.has_start() && self.has_end()
        }

        pub fn get_children(&self) -> Range<usize> {
            self.get_start()
                .zip(self.get_end())
                .map(|(start, end): (usize, usize)| start..end)
                .unwrap_or_default()
        }
    }

    pub struct DirectoryMut<'f, 's: 'f>(&'f mut FileDescriptor<'s>);

    impl<'f, 's: 'f> DirectoryMut<'f, 's> {
        #[inline]
        pub fn set_start(&mut self, start: usize) {
            self.0.start = start;
        }

        pub fn set_end(&mut self, end: usize) {
            if let Some(start) = self.get_start() {
                let len: usize = end - start;

                assert!(len < LEN_MASK, "Length of {len} cannot be stored");
                self.0.len = self.0.type_mask() | len;
            }
        }
    }

    impl<'f, 's: 'f> Deref for DirectoryMut<'f, 's> {
        type Target = Directory<'f, 's>;

        fn deref<'d>(&'d self) -> &'d Self::Target {
            unsafe { transmute(self) }
        }
    }
}

// don't store vecs of lines, just parse as a line, then store/handle accordingly
enum Line<'s> {
    Command(Command<'s>),
    Output(file_descriptor::FileDescriptor<'s>),
}

#[derive(Debug)]
pub enum LineParseError<'s> {
    NoInitialChar,
    FailedToParseCommand(CommandParseError<'s>),
    FailedToParseOutput(file_descriptor::FileDescriptorParseError<'s>),
    ExtraTokenFound(&'s str),
}

impl<'s> TryFrom<&'s str> for Line<'s> {
    type Error = LineParseError<'s>;

    fn try_from(line_str: &'s str) -> Result<Self, Self::Error> {
        use LineParseError as Error;

        let mut token_iter: Split<char> = line_str.split(' ');
        let token_stream: TokenStream<Split<char>> = TokenStream::new(&mut token_iter);

        match line_str.get(..1_usize) {
            None => Err(Error::NoInitialChar),
            Some("$") => Ok(Self::Command(
                token_stream
                    .try_into()
                    .map_err(Error::FailedToParseCommand)?,
            )),
            Some(_) => Ok(Self::Output(
                token_stream
                    .try_into()
                    .map_err(Error::FailedToParseOutput)?,
            )),
        }
        .and_then(|line: Self| -> Result<Self, Error> {
            token_iter
                .next()
                .map(Error::ExtraTokenFound)
                .map(Err)
                .unwrap_or(Ok(line))
        })
    }
}

mod file_system {
    use super::{file_descriptor::*, *};

    /// An error message pertinent to internal processing
    ///
    /// This is likely only relevant to developers, not function callers
    #[derive(Copy, Clone, Debug)]
    pub enum FileSystemInternalError {
        DependentFileDescriptorHasNoSize,
        DirectoryAlreadyInitialized,
        DirectoryUninitialized,
        EmptyPath,
        FileInPath,
        InvalidDirectoryChildrenIndices,
        InvalidFileDescriptorIndex,
    }

    impl FileSystemInternalError {
        fn include_path(self) -> bool {
            !matches!(self, Self::EmptyPath)
        }
    }

    #[derive(Debug, Default)]
    pub struct FileSystem<'s>(pub Vec<FileDescriptor<'s>>);

    impl<'s> FileSystem<'s> {
        fn path_as_strs(&self, path: &Vec<usize>) -> Vec<&'s str> {
            path.iter()
                .map(|index: &usize| -> &'s str {
                    let index: usize = *index;

                    if index < self.0.len() {
                        self.0[index].name
                    } else {
                        "[invalid index]"
                    }
                })
                .collect()
        }

        fn compute_sizes(&mut self) -> Result<(), IError> {
            for file_descriptor_index in (0_usize..self.0.len()).rev() {
                let file_descriptor: &FileDescriptor = &self.0[file_descriptor_index];
                let directory: Directory = match file_descriptor.get_directory() {
                    Some(directory) => directory,
                    None => {
                        continue;
                    }
                };

                if !directory.has_start_and_end() {
                    return Err(IError::InvalidDirectoryChildrenIndices);
                }

                let mut size: usize = 0_usize;

                for child_file_descriptor_index in directory.get_children() {
                    let child_file_descriptor: &FileDescriptor =
                        &self.0[child_file_descriptor_index];

                    match child_file_descriptor.get_size() {
                        Some(child_size) => {
                            size += child_size;
                        }
                        None => Err(IError::DependentFileDescriptorHasNoSize)?,
                    }
                }

                self.0[file_descriptor_index].set_size(size);
            }

            Ok(())
        }
    }

    #[derive(Debug)]
    pub enum FileSystemParseError<'s> {
        FailedToParseLine(LineParseError<'s>),
        InvalidChildDirectory(&'s str),
        ReceivedOutputWhileNotListing,
        InternalError(FileSystemInternalError),
    }

    use FileSystemInternalError as IError;
    use FileSystemParseError as PError;
    type PathOption<'s> = Option<Box<Vec<&'s str>>>;
    pub type Error<'s> = (PError<'s>, PathOption<'s>);

    #[derive(Default)]
    enum IsListing {
        #[default]
        No,
        New,
        Old,
    }

    #[derive(Default)]
    struct ParseState<'s> {
        file_system_str: &'s str,
        file_system: FileSystem<'s>,

        /// Indices of the current path `file_system.0`, which must only contain indices to
        /// `FileDescriptor::Directory` variants
        path: Vec<usize>,
        is_listing: IsListing,
    }

    impl<'s> ParseState<'s> {
        fn new(file_system_str: &'s str) -> Self {
            Self {
                file_system_str,
                ..Default::default()
            }
        }

        fn parse_error(&self, parse_error: PError<'s>) -> Result<(), Error<'s>> {
            Err((parse_error, self.path_option(true)))
        }

        fn internal_error(&self, internal_error: IError) -> Result<(), Error<'s>> {
            Err((
                PError::InternalError(internal_error),
                self.path_option(internal_error.include_path()),
            ))
        }

        fn path_option(&self, include_path: bool) -> PathOption<'s> {
            if include_path {
                Some(Box::new(self.file_system.path_as_strs(&self.path)))
            } else {
                None
            }
        }

        fn pop_dir(&mut self) -> Result<usize, Error<'s>> {
            self.path
                .pop()
                .ok_or_else(|| self.internal_error(IError::EmptyPath).unwrap_err())
        }

        fn push_dir(&mut self, name: &'s str) -> Result<(), Error<'s>> {
            if name == "." {
                self.path.clear();
                self.path.push(0_usize);

                if self.file_system.0.is_empty() {
                    self.file_system.0.push(FileDescriptor::new(name, true));
                }

                Ok(())
            } else {
                let current_directory: Directory = self.get_current_directory()?.1;

                if !current_directory.has_start_and_end() {
                    self.internal_error(IError::DirectoryUninitialized)?;
                }

                let children: Range<usize> = current_directory.get_children();
                let start: usize = children.start;
                let child_directory_index: usize = self
                    .file_system
                    .0
                    .get(children)
                    .ok_or_else(|| {
                        self.internal_error(IError::InvalidDirectoryChildrenIndices)
                            .unwrap_err()
                    })?
                    .iter()
                    .position(|file_descriptor: &FileDescriptor| -> bool {
                        file_descriptor.is_directory() && file_descriptor.name == name
                    })
                    .ok_or_else(|| {
                        self.parse_error(PError::InvalidChildDirectory(name))
                            .unwrap_err()
                    })?
                    + start;

                self.path.push(child_directory_index);

                Ok(())
            }
        }

        fn push_file_descriptor(&mut self, file_descriptor: FileDescriptor<'s>) {
            self.file_system.0.push(file_descriptor)
        }

        fn get_current_file_descriptor_index(&self) -> Result<usize, Error<'s>> {
            self.path
                .last()
                .copied()
                .ok_or_else(|| self.internal_error(IError::EmptyPath).unwrap_err())
        }

        fn get_current_directory<'f>(&'f self) -> Result<(usize, Directory<'f, 's>), Error<'s>> {
            let current_file_descriptor_index: usize = self.get_current_file_descriptor_index()?;

            match self.file_system.0.get(current_file_descriptor_index) {
                None => self
                    .internal_error(IError::InvalidFileDescriptorIndex)
                    .map(unreachable_any),
                Some(file_descriptor) => file_descriptor
                    .get_directory()
                    .map(|directory: Directory| (current_file_descriptor_index, directory))
                    .ok_or_else(|| self.internal_error(IError::FileInPath).unwrap_err()),
            }
        }

        fn get_current_directory_mut<'f>(
            &'f mut self,
        ) -> Result<(usize, DirectoryMut<'f, 's>), Error<'s>> {
            let current_directory_index: usize = self.get_current_directory()?.0;

            Ok((
                current_directory_index,
                self.file_system.0[current_directory_index]
                    .get_directory_mut()
                    .unwrap(),
            ))
        }

        fn set_current_directory_start(&mut self) -> Result<IsListing, Error<'s>> {
            let start: usize = self.file_descriptor_len();
            let mut directory: DirectoryMut = self.get_current_directory_mut()?.1;

            Ok(if !directory.has_start() {
                directory.set_start(start);

                IsListing::New
            } else {
                IsListing::Old
            })
        }

        fn set_current_directory_end(&mut self) -> Result<(), Error<'s>> {
            {
                let end: usize = self.file_descriptor_len();
                let mut directory: DirectoryMut = self.get_current_directory_mut()?.1;

                if directory.has_end() {
                    Err(IError::DirectoryAlreadyInitialized)
                } else {
                    directory.set_end(end);
                    Ok(())
                }
            }
            .map_err(|internal_error: IError| self.internal_error(internal_error).unwrap_err())
        }

        fn file_descriptor_len(&self) -> usize {
            self.file_system.0.len()
        }

        fn parse(mut self) -> Result<FileSystem<'s>, Error<'s>> {
            for line_str in self.file_system_str.split('\n') {
                match Line::try_from(line_str)
                    .map_err(PError::FailedToParseLine)
                    .map_err(|parse_error: PError| -> Error {
                        self.parse_error(parse_error).unwrap_err()
                    })? {
                    Line::Command(command) => {
                        match self.is_listing {
                            IsListing::New => {
                                self.set_current_directory_end()?;
                                self.is_listing = IsListing::No;
                            }
                            IsListing::Old => {
                                self.is_listing = IsListing::No;
                            }
                            _ => {}
                        }

                        match command {
                            Command::ChangeDirectory(ChangeDirectory::Root) => {
                                self.push_dir(".")?;
                            }
                            Command::ChangeDirectory(ChangeDirectory::Parent) => {
                                self.pop_dir()?;
                            }
                            Command::ChangeDirectory(ChangeDirectory::Child(
                                child_directory_str,
                            )) => {
                                self.push_dir(child_directory_str)?;
                            }
                            Command::List => {
                                self.is_listing = self.set_current_directory_start()?;
                            }
                        }
                    }
                    Line::Output(output) => match self.is_listing {
                        IsListing::No => {
                            self.parse_error(PError::ReceivedOutputWhileNotListing)?;
                        }
                        _ => self.push_file_descriptor(output),
                    },
                }
            }

            if matches!(self.is_listing, IsListing::New) {
                self.set_current_directory_end()?;
            }

            self.file_system
                .compute_sizes()
                .map_err(|internal_error: IError| {
                    self.internal_error(internal_error).unwrap_err()
                })?;

            Ok(self.file_system)
        }
    }

    impl<'s> TryFrom<&'s str> for FileSystem<'s> {
        type Error = Error<'s>;

        fn try_from(file_system_str: &'s str) -> Result<Self, Self::Error> {
            ParseState::new(file_system_str).parse()
        }
    }

    pub fn parse<'s>(file_system_str: &'s str) -> Option<FileSystem<'s>> {
        FileSystem::try_from(file_system_str)
            .map_err(|error: Error<'s>| {
                eprintln!("{error:#?}");
            })
            .ok()
    }
}

fn sum_directory_sizes_at_most_n<'s>(file_system: &file_system::FileSystem<'s>, n: usize) -> usize {
    file_system
        .0
        .iter()
        .filter_map(
            |file_descriptor: &file_descriptor::FileDescriptor| -> Option<usize> {
                if file_descriptor.is_directory() {
                    match file_descriptor.get_size() {
                        Some(size) if size <= n => Some(size),
                        _ => None,
                    }
                } else {
                    None
                }
            },
        )
        .sum()
}

fn smallest_directory_size_to_free_n<'s>(
    file_system: &file_system::FileSystem<'s>,
    n: usize,
    total: usize,
) -> usize {
    let unused: usize = match file_system.0.first().and_then(FileDescriptor::get_size) {
        Some(used) => total - used,
        None => {
            // There's no information, so the disk must be empty?
            return 0_usize;
        }
    };
    let enough: usize = n - unused;

    file_system
        .0
        .iter()
        .filter_map(
            |file_descriptor: &file_descriptor::FileDescriptor| -> Option<usize> {
                if file_descriptor.is_directory() {
                    match file_descriptor.get_size() {
                        Some(size) if size >= enough => Some(size),
                        _ => None,
                    }
                } else {
                    None
                }
            },
        )
        .min()
        .unwrap()
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day7.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(input_file_path, |input: &str| {
                match file_system::parse(input) {
                    Some(file_system) => {
                        println!(
                            "sum_directory_sizes_at_most_n == {}\n\
                            smallest_directory_size_to_free_n == {}",
                            sum_directory_sizes_at_most_n(&file_system, 100_000_usize),
                            smallest_directory_size_to_free_n(
                                &file_system,
                                30_000_000_usize,
                                70_000_000_usize
                            )
                        );
                    }
                    None => {
                        panic!();
                    }
                }
            })
        }
    {
        eprintln!(
            "Encountered error {} when opening file \"{}\"",
            err, input_file_path
        );
    }
}

#[test]
fn test() {
    use super::{file_descriptor::*, file_system::*, *};
    const FILE_SYSTEM_STR: &str = "$ cd /\n\
        $ ls\n\
        dir a\n\
        14848514 b.txt\n\
        8504156 c.dat\n\
        dir d\n\
        $ cd a\n\
        $ ls\n\
        dir e\n\
        29116 f\n\
        2557 g\n\
        62596 h.lst\n\
        $ cd e\n\
        $ ls\n\
        584 i\n\
        $ cd ..\n\
        $ cd ..\n\
        $ cd d\n\
        $ ls\n\
        4060174 j\n\
        8033020 d.log\n\
        5626152 d.ext\n\
        7214296 k";

    match parse(FILE_SYSTEM_STR) {
        Some(file_system) => {
            assert_eq!(
                sum_directory_sizes_at_most_n(&file_system, 100_000_usize),
                95_437_usize
            );
            assert_eq!(
                smallest_directory_size_to_free_n(&file_system, 30_000_000_usize, 70_000_000_usize),
                24_933_642_usize
            );
        }
        None => {
            panic!();
        }
    }
}
