use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{alpha1, line_ending},
        combinator::{map, map_opt, opt},
        error::Error,
        multi::many0,
        sequence::{preceded, terminated, tuple},
        Err, IResult,
    },
    std::{collections::HashMap, str::from_utf8_unchecked},
};

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
enum StepType {
    Equals(u8),
    Minus,
}

impl StepType {
    const BYTES: [u8; 2_usize] = [b'=', b'-'];
    const MAX_BYTES_LEN: usize = StepType::Equals(0_u8).bytes_len();
    const MAX_VALUE: u8 = 9_u8;
    const VALUE_OFFSET: u8 = b'0';

    fn byte(self) -> u8 {
        self.byte_tag().as_bytes()[0_usize]
    }

    fn byte_tag(self) -> &'static str {
        // SAFETY: `Self::BYTES` is a const slice of two ASCII characters.
        unsafe {
            from_utf8_unchecked(
                &Self::BYTES[match self {
                    StepType::Equals(_) => 0_usize..1_usize,
                    StepType::Minus => 1_usize..2_usize,
                }],
            )
        }
    }

    const fn bytes_len(self) -> usize {
        match self {
            StepType::Equals(_) => 2_usize,
            StepType::Minus => 1_usize,
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Step {
    bytes: [u8; Step::BYTE_COUNT],
    len: u8,
    step_type: StepType,
}

type Label = [u8; Step::MAX_LABEL_LEN];

impl Step {
    const BYTE_COUNT: usize = 8_usize;
    const MAX_LABEL_LEN: usize = Self::BYTE_COUNT - StepType::MAX_BYTES_LEN;

    fn append_label(&mut self, label: &str) -> Option<()> {
        if label.len() > Self::MAX_LABEL_LEN {
            None
        } else {
            self.bytes[..label.len()].copy_from_slice(label.as_bytes());
            self.len = label.len() as u8;

            Some(())
        }
    }

    fn append_step_type(&mut self) -> Option<()> {
        let len: usize = self.len as usize;

        self.bytes[len] = self.step_type.byte();

        match self.step_type {
            StepType::Equals(value) => {
                if value > StepType::MAX_VALUE {
                    None?;
                } else {
                    self.bytes[len + 1_usize] = value + StepType::VALUE_OFFSET;
                }
            }
            _ => (),
        }

        self.len = (len + self.step_type.bytes_len()) as u8;

        Some(())
    }

    fn try_from_label_and_step_type(label: &str, step_type: StepType) -> Option<Self> {
        let mut step: Self = Self {
            bytes: [0_u8; Self::BYTE_COUNT],
            len: 0_u8,
            step_type,
        };

        step.append_label(label)?;
        step.append_step_type()?;

        Some(step)
    }

    fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    fn label_bytes(&self) -> &[u8] {
        &self.bytes()[..self.len as usize - self.step_type.bytes_len()]
    }

    fn label(&self) -> Label {
        let label_bytes: &[u8] = self.label_bytes();

        let mut label: Label = Label::default();

        label[..label_bytes.len()].copy_from_slice(label_bytes);

        label
    }
}

impl Parse for Step {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(
            tuple((
                alpha1,
                alt((
                    map(
                        preceded(tag(StepType::Equals(0_u8).byte_tag()), parse_integer::<u8>),
                        StepType::Equals,
                    ),
                    map(tag(StepType::Minus.byte_tag()), |_| StepType::Minus),
                )),
            )),
            |(label, step_type)| Step::try_from_label_and_step_type(label, step_type),
        )(input)
    }
}

impl TryFrom<(&str, StepType)> for Step {
    type Error = ();

    fn try_from((label, step_type): (&str, StepType)) -> Result<Self, Self::Error> {
        Self::try_from_label_and_step_type(label, step_type).ok_or(())
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Lens {
    label: Label,
    focal_length: u8,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
pub struct Solution(Vec<Step>);

impl Solution {
    const MULTIPLICATION_FACTOR: u16 = 17_u16;
    const REMAINDER_FACTOR: u16 = u8::MAX as u16 + 1_u16;

    fn holiday_ascii_string_helper_algorithm(bytes: &[u8]) -> u8 {
        bytes.iter().copied().fold(0_u16, |current_value, byte| {
            (current_value + byte as u16) * Self::MULTIPLICATION_FACTOR % Self::REMAINDER_FACTOR
        }) as u8
    }

    fn iter_step_bytes(&self) -> impl Iterator<Item = &[u8]> + '_ {
        self.0.iter().map(Step::bytes)
    }

    fn iter_step_hashes(&self) -> impl Iterator<Item = u8> + '_ {
        self.iter_step_bytes()
            .map(Self::holiday_ascii_string_helper_algorithm)
    }

    fn sum_step_hashes(&self) -> u32 {
        self.iter_step_hashes().map(|hash| hash as u32).sum()
    }

    fn holiday_ascii_string_helper_manual_arrangement_procedure(&self) -> HashMap<u8, Vec<Lens>> {
        let mut hashmap: HashMap<u8, Vec<Lens>> = HashMap::new();

        for step in self.0.iter() {
            let box_key: u8 = Self::holiday_ascii_string_helper_algorithm(step.label_bytes());
            let mut box_lenses: Option<&mut Vec<Lens>> = hashmap.get_mut(&box_key);

            // Only construct the `Vec` if it doesn't already exist and it won't be empty after this
            // step.
            if box_lenses.is_none() && !matches!(step.step_type, StepType::Minus) {
                hashmap.insert(box_key, Vec::new());

                box_lenses = hashmap.get_mut(&box_key);
            }

            if let Some(box_lenses) = box_lenses {
                let label: Label = step.label();

                match step.step_type {
                    StepType::Equals(focal_length) => {
                        if let Some(lens) = box_lenses.iter_mut().find(|lens| lens.label == label) {
                            lens.focal_length = focal_length;
                        } else {
                            box_lenses.push(Lens {
                                label,
                                focal_length,
                            });
                        }
                    }
                    StepType::Minus => {
                        if let Some(index) = box_lenses.iter().position(|lens| lens.label == label)
                        {
                            box_lenses.remove(index);
                        }
                    }
                }
            }
        }

        hashmap
    }

    fn iter_hashmap_focusing_powers(&self) -> impl Iterator<Item = u32> {
        self.holiday_ascii_string_helper_manual_arrangement_procedure()
            .into_iter()
            .flat_map(|(box_key, lenses)| {
                lenses.into_iter().enumerate().map(move |(index, lens)| {
                    (box_key as u32 + 1_u32) * (index as u32 + 1_u32) * lens.focal_length as u32
                })
            })
    }

    fn sum_hashmap_focusing_powers(&self) -> u32 {
        self.iter_hashmap_focusing_powers().sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            many0(terminated(Step::parse, opt(alt((tag(","), line_ending))))),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_step_hashes());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_hashmap_focusing_powers());
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
    use {
        super::*,
        std::{collections::HashSet, sync::OnceLock},
    };

    const SOLUTION_STR: &'static str = "rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7";

    fn solution() -> &'static Solution {
        use StepType::{Equals as E, Minus as M};

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        let step = |label: &str, step_type: StepType| Step::try_from((label, step_type)).unwrap();

        ONCE_LOCK.get_or_init(|| {
            Solution(vec![
                step("rn", E(1_u8)),
                step("cm", M),
                step("qp", E(3_u8)),
                step("cm", E(2_u8)),
                step("qp", M),
                step("pc", E(4_u8)),
                step("ot", E(9_u8)),
                step("ab", E(5_u8)),
                step("pc", M),
                step("pc", E(6_u8)),
                step("ot", E(7_u8)),
            ])
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_holiday_ascii_string_helper_algorithm() {
        assert_eq!(
            Solution::holiday_ascii_string_helper_algorithm(b"HASH"),
            52_u8
        );
    }

    #[test]
    fn test_iter_step_hashes() {
        assert_eq!(
            solution().iter_step_hashes().collect::<Vec<u8>>(),
            vec![30_u8, 253_u8, 97_u8, 47_u8, 14_u8, 180_u8, 9_u8, 197_u8, 48_u8, 214_u8, 231_u8]
        );
    }

    #[test]
    fn test_sum_step_hashes() {
        assert_eq!(solution().sum_step_hashes(), 1320_u32);
    }

    #[test]
    fn test_holiday_ascii_string_helper_manual_arrangement_procedure() {
        let lens = |label_str: &str, focal_length: u8| {
            let mut label: Label = Label::default();

            label[..label_str.len()].copy_from_slice(label_str.as_bytes());

            Lens {
                label,
                focal_length,
            }
        };

        assert_eq!(
            solution().holiday_ascii_string_helper_manual_arrangement_procedure(),
            [
                (0_u8, vec![lens("rn", 1_u8), lens("cm", 2_u8)]),
                (1_u8, vec![]),
                (
                    3_u8,
                    vec![lens("ot", 7_u8), lens("ab", 5_u8), lens("pc", 6_u8)]
                )
            ]
            .into_iter()
            .collect::<HashMap<u8, Vec<Lens>>>()
        );
    }

    #[test]
    fn test_iter_hashmap_focusing_powers() {
        assert_eq!(
            solution()
                .iter_hashmap_focusing_powers()
                .collect::<HashSet<u32>>(),
            [1_u32, 4_u32, 28_u32, 40_u32, 72_u32]
                .into_iter()
                .collect::<HashSet<u32>>()
        );
    }

    #[test]
    fn test_sum_hashmap_focusing_powers() {
        assert_eq!(solution().sum_hashmap_focusing_powers(), 145_u32);
    }
}
