use {aoc_2022::*, clap::Parser, std::collections::VecDeque};

#[derive(Default)]
struct ProcessState {
    queue: VecDeque<u8>,
    processed: usize,
    present_by_letter: [u8; 26_usize],
    present_goal: u8,
    present_total: u8,
}

const A_OFFSET: u8 = b'a' as u8;

#[derive(Debug)]
enum ProcessCharError {
    CharIsNotAsciiLowercase(char),
    PoppingCharWithZeroCount(char),
    DecrementingCharCountToZeroWithZeroPresentTotal(char),
}

#[derive(Debug)]
enum DetectMarkerError {
    ConsecutiveDistinctCharsIsTooLarge,
    ProcessCharError(ProcessCharError),
    NoMarkerFound(usize),
}

impl ProcessState {
    fn process_char(&mut self, new_char: char) -> Result<(), ProcessCharError> {
        use ProcessCharError as Error;

        if !new_char.is_ascii_lowercase() {
            return Err(Error::CharIsNotAsciiLowercase(new_char));
        }

        self.processed += 1_usize;

        let new_data_byte: u8 = new_char as u8 - A_OFFSET;

        self.queue.push_back(new_data_byte);

        let new_char_present: &mut u8 = &mut self.present_by_letter[new_data_byte as usize];

        if *new_char_present == 0_u8 {
            self.present_total += 1_u8;
        }

        *new_char_present += 1_u8;

        if self.queue.len() > self.present_goal as usize {
            let old_data_byte: u8 = self.queue.pop_front().unwrap();
            let old_char_present: &mut u8 = &mut self.present_by_letter[old_data_byte as usize];

            if *old_char_present == 0_u8 {
                return Err(Error::PoppingCharWithZeroCount(
                    (old_data_byte + A_OFFSET) as char,
                ));
            }

            *old_char_present -= 1_u8;

            if *old_char_present == 0_u8 {
                if self.present_total == 0_u8 {
                    return Err(Error::DecrementingCharCountToZeroWithZeroPresentTotal(
                        (old_data_byte + A_OFFSET) as char,
                    ));
                }

                self.present_total -= 1_u8;
            }
        }

        Ok(())
    }

    fn has_reached_goal(&self) -> bool {
        self.present_total >= self.present_goal
    }

    fn detect_marker(mut self, datastream: &str) -> Result<usize, DetectMarkerError> {
        use DetectMarkerError as Error;

        for new_char in datastream.chars() {
            self.process_char(new_char)
                .map_err(Error::ProcessCharError)?;

            if self.has_reached_goal() {
                return Ok(self.processed);
            }
        }

        Err(Error::NoMarkerFound(self.processed))
    }
}

struct ConsecutiveDistinctChars(usize);

impl TryFrom<ConsecutiveDistinctChars> for ProcessState {
    type Error = DetectMarkerError;

    fn try_from(consecutive_distinct_chars: ConsecutiveDistinctChars) -> Result<Self, Self::Error> {
        use DetectMarkerError as Error;

        if consecutive_distinct_chars.0 > 26_usize {
            Err(Error::ConsecutiveDistinctCharsIsTooLarge)
        } else {
            Ok(Self {
                queue: VecDeque::with_capacity(consecutive_distinct_chars.0 + 1_usize),
                present_goal: consecutive_distinct_chars.0 as u8,
                ..ProcessState::default()
            })
        }
    }
}

fn detect_marker(
    datastream: &str,
    consecutive_distinct_chars: usize,
) -> Result<usize, DetectMarkerError> {
    ProcessState::try_from(ConsecutiveDistinctChars(consecutive_distinct_chars))?
        .detect_marker(datastream)
}

fn detect_start_of_packet_marker(datastream: &str) -> Result<usize, DetectMarkerError> {
    detect_marker(datastream, 4_usize)
}

fn detect_start_of_message_marker(datastream: &str) -> Result<usize, DetectMarkerError> {
    detect_marker(datastream, 14_usize)
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day6.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(input_file_path, |input: &str| {
                println!(
                    "detect_start_of_packet_marker == {:#?}\n\
                    detect_start_of_message_marker == {:#?}",
                    detect_start_of_packet_marker(input),
                    detect_start_of_message_marker(input)
                );
            })
        }
    {
        eprintln!(
            "Encountered error {} when opening file \"{}\"",
            err, input_file_path
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_start_of_packet_marker() {
        for (datastream, expected_start_of_packet, expected_start_of_message) in [
            ("mjqjpqmgbljsphdztnvjfqwrcgsmlb", 7_usize, 19_usize),
            ("bvwbjplbgvbhsrlpgdmjqwftvncz", 5_usize, 23_usize),
            ("nppdvjthqldpwncqszvftbrmjlhg", 6_usize, 23_usize),
            ("nznrnfrfntjfmvfwmzdfjlvtqnbhcprsg", 10_usize, 29_usize),
            ("zcfzfwzzqfrljwzlrfnpqdbhtmscgvjw", 11_usize, 26_usize),
        ] {
            assert!(matches!(
                detect_start_of_packet_marker(datastream),
                Ok(start_of_packet) if start_of_packet == expected_start_of_packet
            ));
            assert!(matches!(
                detect_start_of_message_marker(datastream),
                Ok(start_of_message) if start_of_message == expected_start_of_message
            ));
        }
    }
}
