use {
    crate::*,
    nom::{
        character::complete::satisfy,
        combinator::{map, map_opt, map_res, opt, success},
        error::Error,
        multi::many0,
        sequence::tuple,
        Err, IResult,
    },
    std::iter::{from_fn, repeat},
};

/* --- Day 9: Disk Fragmenter ---

Another push of the button leaves you in the familiar hallways of some friendly amphipods! Good thing you each somehow got your own personal mini submarine. The Historians jet away in search of the Chief, mostly by driving directly into walls.

While The Historians quickly figure out how to pilot these things, you notice an amphipod in the corner struggling with his computer. He's trying to make more contiguous free space by compacting all of the files, but his program isn't working; you offer to help.

He shows you the disk map (your puzzle input) he's already generated. For example:

2333133121414131402

The disk map uses a dense format to represent the layout of files and free space on the disk. The digits alternate between indicating the length of a file and the length of free space.

So, a disk map like 12345 would represent a one-block file, two blocks of free space, a three-block file, four blocks of free space, and then a five-block file. A disk map like 90909 would represent three nine-block files in a row (with no free space between them).

Each file on disk also has an ID number based on the order of the files as they appear before they are rearranged, starting with ID 0. So, the disk map 12345 has three files: a one-block file with ID 0, a three-block file with ID 1, and a five-block file with ID 2. Using one character for each block where digits are the file ID and . is free space, the disk map 12345 represents these individual blocks:

0..111....22222

The first example above, 2333133121414131402, represents these individual blocks:

00...111...2...333.44.5555.6666.777.888899

The amphipod would like to move file blocks one at a time from the end of the disk to the leftmost free space block (until there are no gaps remaining between file blocks). For the disk map 12345, the process looks like this:

0..111....22222
02.111....2222.
022111....222..
0221112...22...
02211122..2....
022111222......

The first example requires a few more steps:

00...111...2...333.44.5555.6666.777.888899
009..111...2...333.44.5555.6666.777.88889.
0099.111...2...333.44.5555.6666.777.8888..
00998111...2...333.44.5555.6666.777.888...
009981118..2...333.44.5555.6666.777.88....
0099811188.2...333.44.5555.6666.777.8.....
009981118882...333.44.5555.6666.777.......
0099811188827..333.44.5555.6666.77........
00998111888277.333.44.5555.6666.7.........
009981118882777333.44.5555.6666...........
009981118882777333644.5555.666............
00998111888277733364465555.66.............
0099811188827773336446555566..............

The final step of this file-compacting process is to update the filesystem checksum. To calculate the checksum, add up the result of multiplying each of these blocks' position with the file ID number it contains. The leftmost block is in position 0. If a block contains free space, skip it instead.

Continuing the first example, the first few blocks' position multiplied by its file ID number are 0 * 0 = 0, 1 * 0 = 0, 2 * 9 = 18, 3 * 9 = 27, 4 * 8 = 32, and so on. In this example, the checksum is the sum of these, 1928.

Compact the amphipod's hard drive using the process he requested. What is the resulting filesystem checksum? (Be careful copy/pasting the input for this puzzle; it is a single, very long line.)

--- Part Two ---

Upon completion, two things immediately become clear. First, the disk definitely has a lot more contiguous free space, just like the amphipod hoped. Second, the computer is running much more slowly! Maybe introducing all of that file system fragmentation was a bad idea?

The eager amphipod already has a new plan: rather than move individual blocks, he'd like to try compacting the files on his disk by moving whole files instead.

This time, attempt to move whole files to the leftmost span of free space blocks that could fit the file. Attempt to move each file exactly once in order of decreasing file ID number starting with the file with the highest file ID number. If there is no span of free space to the left of a file that is large enough to fit the file, the file does not move.

The first example from above now proceeds differently:

00...111...2...333.44.5555.6666.777.888899
0099.111...2...333.44.5555.6666.777.8888..
0099.1117772...333.44.5555.6666.....8888..
0099.111777244.333....5555.6666.....8888..
00992111777.44.333....5555.6666.....8888..

The process of updating the filesystem checksum is the same; now, this example's checksum would be 2858.

Start over, now compacting the amphipod's hard drive using this new method instead. What is the resulting filesystem checksum? */

type FileIndexRaw = u16;
type FileIndex = Index<FileIndexRaw>;

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct File {
    id: u16,
    file_block_len: u8,
    free_block_len: u8,
    prev_file_index: FileIndex,
    next_file_index: FileIndex,
}

impl File {
    const INVALID_ID: u16 = u16::MAX;

    const fn invalid() -> Self {
        Self {
            id: Self::INVALID_ID,
            file_block_len: 0_u8,
            free_block_len: 0_u8,
            prev_file_index: FileIndex::invalid(),
            next_file_index: FileIndex::invalid(),
        }
    }

    fn parse_block_len<'i>(input: &'i str) -> IResult<&'i str, u8> {
        map(satisfy(|c| c.is_ascii_digit()), |c| c as u8 - b'0')(input)
    }
}

impl Default for File {
    fn default() -> Self {
        Self::invalid()
    }
}

impl Parse for File {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(
            tuple((
                Self::parse_block_len,
                map(opt(Self::parse_block_len), Option::unwrap_or_default),
            )),
            |(file_block_len, free_block_len)| {
                (file_block_len != 0_u8).then(|| Self {
                    file_block_len,
                    free_block_len,
                    ..Default::default()
                })
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct DiskMap {
    files: Vec<File>,
    file_block_tail_file_index: FileIndex,
    free_block_head_file_index: FileIndex,
}

impl DiskMap {
    fn iter_block_file_ids(&self) -> impl Iterator<Item = u16> + '_ {
        let mut file_index: FileIndex = (self.files.len() > 0_usize)
            .then_some(0_usize.into())
            .unwrap_or_default();

        from_fn(move || {
            file_index.is_valid().then(|| {
                let file: File = self.files[file_index.get()];

                file_index = file.next_file_index;

                file
            })
        })
        .flat_map(|file| {
            repeat(file.id)
                .take(file.file_block_len as usize)
                .chain(repeat(File::INVALID_ID).take(file.free_block_len as usize))
        })
    }

    fn checksum(&self) -> u64 {
        self.iter_block_file_ids()
            .enumerate()
            .map(|(index, file_id)| {
                (file_id != File::INVALID_ID)
                    .then(|| index as u64 * file_id as u64)
                    .unwrap_or_default()
            })
            .sum()
    }

    fn update_file_block_tail_file_index(&mut self) {
        let mut file_block_tail_file_index: FileIndex = self.file_block_tail_file_index;
        let mut file_block_next_file_index: FileIndex;

        while file_block_tail_file_index.is_valid() && {
            let file: File = self.files[file_block_tail_file_index.get()];

            file_block_next_file_index = file.next_file_index;

            file_block_next_file_index.is_valid()
        } {
            file_block_tail_file_index = file_block_next_file_index;
        }

        self.file_block_tail_file_index = file_block_tail_file_index;
    }

    fn update_free_block_head_file_index(&mut self) {
        let mut free_block_head_file_index: FileIndex = self.free_block_head_file_index;
        let mut free_block_next_file_index: FileIndex;

        while free_block_head_file_index.is_valid() && {
            let file: File = self.files[free_block_head_file_index.get()];

            free_block_next_file_index = file.next_file_index;

            file.free_block_len == 0_u8
        } {
            free_block_head_file_index = free_block_next_file_index;
        }

        self.free_block_head_file_index = free_block_head_file_index;
    }

    fn compact_1(&mut self) {
        while self.free_block_head_file_index.is_valid()
            && self.file_block_tail_file_index.is_valid()
        {
            let src_file_file: File = self.files[self.file_block_tail_file_index.get()];
            let dst_free_file: File = self.files[self.free_block_head_file_index.get()];

            if self.file_block_tail_file_index == dst_free_file.next_file_index {
                assert_eq!(
                    self.free_block_head_file_index,
                    src_file_file.prev_file_index
                );
                assert_eq!(src_file_file.free_block_len, 0_u8);

                self.files[self.free_block_head_file_index.get()].free_block_len = 0_u8;
                self.free_block_head_file_index = FileIndex::invalid();
            } else {
                let src_file_block_len: u8 = src_file_file.file_block_len;
                let dst_free_block_len: u8 = dst_free_file.free_block_len;
                let transfer_block_len: u8 = src_file_block_len.min(dst_free_block_len);

                assert!(transfer_block_len != 0_u8);

                if transfer_block_len == src_file_block_len {
                    self.files[self.file_block_tail_file_index.get()] = File::invalid();
                    self.file_block_tail_file_index = src_file_file.prev_file_index;

                    if let Some(file_block_tail_file_index) = self.file_block_tail_file_index.opt()
                    {
                        let file: &mut File = &mut self.files[file_block_tail_file_index.get()];

                        file.free_block_len = 0_u8;
                        file.next_file_index = FileIndex::invalid();
                    }
                } else {
                    self.files[self.file_block_tail_file_index.get()].file_block_len -=
                        transfer_block_len;
                }

                {
                    let dst_next_file_index: FileIndex = self.files.len().into();
                    let dst_free_file: &mut File =
                        &mut self.files[self.free_block_head_file_index.get()];

                    dst_free_file.free_block_len = 0_u8;
                    dst_free_file.next_file_index = dst_next_file_index;
                }

                self.files.push(File {
                    id: src_file_file.id,
                    file_block_len: transfer_block_len,
                    free_block_len: dst_free_block_len - transfer_block_len,
                    prev_file_index: self.free_block_head_file_index,
                    next_file_index: dst_free_file.next_file_index,
                });

                self.update_file_block_tail_file_index();
                self.update_free_block_head_file_index();
            }
        }
    }

    fn compact_2(&mut self) {
        // let mut src_file_block_len_to_dst_file_index: [FileIndex; 10_usize] =
        //     [0_usize.into(); 10_usize];

        for src_file_index in (0_usize..self.files.len()).rev() {
            let src_file: File = self.files[src_file_index];
            let src_file_block_len: u8 = src_file.file_block_len;
            let mut dst_file_index: FileIndex = 0_usize.into();
            // src_file_block_len_to_dst_file_index[src_file_block_len as usize];

            if dst_file_index.is_valid() {
                let mut next_dst_file_index: FileIndex;

                while dst_file_index.get() != src_file_index && {
                    let dst_file: File = self.files[dst_file_index.get()];

                    next_dst_file_index = dst_file.next_file_index;

                    dst_file.free_block_len < src_file_block_len
                } {
                    dst_file_index = next_dst_file_index;
                }

                if dst_file_index.get() != src_file_index {
                    let new_next_file_index: FileIndex = self.files.len().into();
                    let dst_file: &mut File = &mut self.files[dst_file_index.get()];
                    let dst_free_block_len: u8 = dst_file.free_block_len;
                    let old_next_file_index: FileIndex = dst_file.next_file_index;

                    dst_file.free_block_len = 0_u8;
                    dst_file.next_file_index = new_next_file_index;

                    self.files.push(File {
                        id: src_file.id,
                        file_block_len: src_file_block_len,
                        free_block_len: dst_free_block_len - src_file_block_len,
                        prev_file_index: dst_file_index,
                        next_file_index: old_next_file_index,
                    });
                    self.files[old_next_file_index.get()].prev_file_index = new_next_file_index;

                    let src_file: &mut File = &mut self.files[src_file_index];

                    src_file.free_block_len += src_file.file_block_len;
                    src_file.file_block_len = 0_u8;

                    // src_file_block_len_to_dst_file_index[src_file_block_len as usize] =
                    //     new_next_file_index;
                } else {
                    // src_file_block_len_to_dst_file_index[src_file_block_len as usize] =
                    //     FileIndex::invalid();
                }
            }
        }
    }
}

impl Parse for DiskMap {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (input, files): (&str, Vec<File>) = many0(File::parse)(input)?;
        let files_len: usize = files.len();
        let mut disk_map: Self = Self {
            files,
            file_block_tail_file_index: files_len
                .checked_sub(1_usize)
                .map_or_else(FileIndex::invalid, FileIndex::from),
            free_block_head_file_index: (files_len > 0_usize)
                .then_some(0_usize.into())
                .unwrap_or_default(),
        };

        for (file_index, file) in disk_map.files.iter_mut().enumerate() {
            file.id = map_res(success(()), |_| file_index.try_into())(input)?.1;

            if let Some(prev_file_index) = file_index.checked_sub(1_usize) {
                file.prev_file_index = prev_file_index.into();
            }

            let next_file_index: usize = file_index + 1_usize;

            if next_file_index < files_len {
                file.next_file_index = next_file_index.into();
            }
        }

        disk_map.update_free_block_head_file_index();

        Ok((input, disk_map))
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(DiskMap);

impl Solution {
    fn compacted_checksum_1(&self) -> u64 {
        let mut disk_map: DiskMap = self.0.clone();

        disk_map.compact_1();

        disk_map.checksum()
    }

    fn compacted_checksum_2(&self) -> u64 {
        let mut disk_map: DiskMap = self.0.clone();

        disk_map.compact_2();

        disk_map.checksum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(DiskMap::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    /// I have doubts that linked list was the correct way to do this.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.compacted_checksum_1());
    }

    /// I'm tired.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.compacted_checksum_2());
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self::parse(input)?.1)
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const SOLUTION_STRS: &'static [&'static str] = &["12345", "2333133121414131402"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(DiskMap {
                    files: vec![
                        File {
                            id: 0_u16,
                            file_block_len: 1_u8,
                            free_block_len: 2_u8,
                            prev_file_index: FileIndex::invalid(),
                            next_file_index: 1_usize.into(),
                        },
                        File {
                            id: 1_u16,
                            file_block_len: 3_u8,
                            free_block_len: 4_u8,
                            prev_file_index: 0_usize.into(),
                            next_file_index: 2_usize.into(),
                        },
                        File {
                            id: 2_u16,
                            file_block_len: 5_u8,
                            free_block_len: 0_u8,
                            prev_file_index: 1_usize.into(),
                            next_file_index: FileIndex::invalid(),
                        },
                    ],
                    file_block_tail_file_index: 2_usize.into(),
                    free_block_head_file_index: 0_usize.into(),
                }),
                Solution(DiskMap {
                    files: vec![
                        File {
                            id: 0_u16,
                            file_block_len: 2_u8,
                            free_block_len: 3_u8,
                            prev_file_index: FileIndex::invalid(),
                            next_file_index: 1_usize.into(),
                        },
                        File {
                            id: 1_u16,
                            file_block_len: 3_u8,
                            free_block_len: 3_u8,
                            prev_file_index: 0_usize.into(),
                            next_file_index: 2_usize.into(),
                        },
                        File {
                            id: 2_u16,
                            file_block_len: 1_u8,
                            free_block_len: 3_u8,
                            prev_file_index: 1_usize.into(),
                            next_file_index: 3_usize.into(),
                        },
                        File {
                            id: 3_u16,
                            file_block_len: 3_u8,
                            free_block_len: 1_u8,
                            prev_file_index: 2_usize.into(),
                            next_file_index: 4_usize.into(),
                        },
                        File {
                            id: 4_u16,
                            file_block_len: 2_u8,
                            free_block_len: 1_u8,
                            prev_file_index: 3_usize.into(),
                            next_file_index: 5_usize.into(),
                        },
                        File {
                            id: 5_u16,
                            file_block_len: 4_u8,
                            free_block_len: 1_u8,
                            prev_file_index: 4_usize.into(),
                            next_file_index: 6_usize.into(),
                        },
                        File {
                            id: 6_u16,
                            file_block_len: 4_u8,
                            free_block_len: 1_u8,
                            prev_file_index: 5_usize.into(),
                            next_file_index: 7_usize.into(),
                        },
                        File {
                            id: 7_u16,
                            file_block_len: 3_u8,
                            free_block_len: 1_u8,
                            prev_file_index: 6_usize.into(),
                            next_file_index: 8_usize.into(),
                        },
                        File {
                            id: 8_u16,
                            file_block_len: 4_u8,
                            free_block_len: 0_u8,
                            prev_file_index: 7_usize.into(),
                            next_file_index: 9_usize.into(),
                        },
                        File {
                            id: 9_u16,
                            file_block_len: 2_u8,
                            free_block_len: 0_u8,
                            prev_file_index: 8_usize.into(),
                            next_file_index: FileIndex::invalid(),
                        },
                    ],
                    file_block_tail_file_index: 9_usize.into(),
                    free_block_head_file_index: 0_usize.into(),
                }),
            ]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for (index, solution_str) in SOLUTION_STRS.iter().copied().enumerate() {
            assert_eq!(
                Solution::try_from(solution_str).as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_iter_block_file_ids() {
        const I: u16 = File::INVALID_ID;

        for (index, block_file_ids) in [
            vec![
                0_u16, I, I, 1_u16, 1_u16, 1_u16, I, I, I, I, 2_u16, 2_u16, 2_u16, 2_u16, 2_u16,
            ],
            vec![
                0_u16, 0_u16, I, I, I, 1_u16, 1_u16, 1_u16, I, I, I, 2_u16, I, I, I, 3_u16, 3_u16,
                3_u16, I, 4_u16, 4_u16, I, 5_u16, 5_u16, 5_u16, 5_u16, I, 6_u16, 6_u16, 6_u16,
                6_u16, I, 7_u16, 7_u16, 7_u16, I, 8_u16, 8_u16, 8_u16, 8_u16, 9_u16, 9_u16,
            ],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index)
                    .0
                    .iter_block_file_ids()
                    .collect::<Vec<u16>>(),
                block_file_ids
            );
        }
    }

    #[test]
    fn test_compact_1() {
        for (index, block_file_ids) in [
            vec![
                0_u16, 2_u16, 2_u16, 1_u16, 1_u16, 1_u16, 2_u16, 2_u16, 2_u16,
            ],
            vec![
                0_u16, 0_u16, 9_u16, 9_u16, 8_u16, 1_u16, 1_u16, 1_u16, 8_u16, 8_u16, 8_u16, 2_u16,
                7_u16, 7_u16, 7_u16, 3_u16, 3_u16, 3_u16, 6_u16, 4_u16, 4_u16, 6_u16, 5_u16, 5_u16,
                5_u16, 5_u16, 6_u16, 6_u16,
            ],
        ]
        .into_iter()
        .enumerate()
        {
            let mut disk_map: DiskMap = solution(index).0.clone();

            disk_map.compact_1();

            assert_eq!(
                disk_map.iter_block_file_ids().collect::<Vec<u16>>(),
                block_file_ids
            );
        }
    }

    #[test]
    fn test_compacted_checksum_1() {
        for (index, compacted_checksum_1) in [
            1_u64 * (3_u64 + 4_u64 + 5_u64) + 2_u64 * (1_u64 + 2_u64 + 6_u64 + 7_u64 + 8_u64),
            1928_u64,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).compacted_checksum_1(), compacted_checksum_1);
        }
    }

    #[test]
    fn test_compact_2() {
        const I: u16 = File::INVALID_ID;

        for (index, block_file_ids) in [
            vec![
                0_u16, I, I, 1_u16, 1_u16, 1_u16, I, I, I, I, 2_u16, 2_u16, 2_u16, 2_u16, 2_u16,
            ],
            vec![
                0_u16, 0_u16, 9_u16, 9_u16, 2_u16, 1_u16, 1_u16, 1_u16, 7_u16, 7_u16, 7_u16, I,
                4_u16, 4_u16, I, 3_u16, 3_u16, 3_u16, I, I, I, I, 5_u16, 5_u16, 5_u16, 5_u16, I,
                6_u16, 6_u16, 6_u16, 6_u16, I, I, I, I, I, 8_u16, 8_u16, 8_u16, 8_u16, I, I,
            ],
        ]
        .into_iter()
        .enumerate()
        {
            let mut disk_map: DiskMap = solution(index).0.clone();

            disk_map.compact_2();

            assert_eq!(
                disk_map.iter_block_file_ids().collect::<Vec<u16>>(),
                block_file_ids
            );
        }
    }

    #[test]
    fn test_compacted_checksum_2() {
        for (index, compacted_checksum_2) in [None, Some(2858_u64)].into_iter().enumerate() {
            if let Some(compacted_checksum_2) = compacted_checksum_2 {
                assert_eq!(solution(index).compacted_checksum_2(), compacted_checksum_2);
            }
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
