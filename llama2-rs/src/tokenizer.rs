use std::{error::Error, path::Path};

use sentencepiece::{PieceWithId, SentencePieceProcessor};

pub struct Tokenizer {
    pub spp: SentencePieceProcessor,
}

impl Tokenizer {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let spp = SentencePieceProcessor::open(path)?;
        Ok(Self { spp })
    }

    fn _encode(&self, input: &str) -> Vec<PieceWithId> {
        self.spp.encode(input).unwrap()
    }

    pub fn is_bos(&self, id: u32) -> bool {
        self.spp.bos_id() == Some(id)
    }

    pub fn encode(&self, input: &str) -> Vec<String> {
        self._encode(input)
            .into_iter()
            .map(|p| p.piece)
            .collect::<Vec<_>>()
    }

    pub fn encode2id(&self, input: &str) -> Vec<u32> {
        self._encode(input)
            .into_iter()
            .map(|p| p.id)
            .collect::<Vec<_>>()
    }

    pub fn decode_id(&self, id: u32) -> String {
        self.spp.decode_piece_ids(&[id]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode() {
        let tokenizer = Tokenizer::new("./tokenizer.model").unwrap();
        let s = "hello world! :)";
        assert_eq!(tokenizer.encode("hello world"), vec!["▁hello", "▁world"]);
        assert_eq!(tokenizer.encode2id("hello world"), vec![22172, 3186]);
        assert_eq!(tokenizer.decode_id(22172), "hello");
        assert_eq!(tokenizer.decode_id(3186), "world");
        assert_eq!(
            tokenizer
                .encode2id(s)
                .iter()
                .map(|id| tokenizer.decode_id(*id))
                .collect::<String>(),
            s.chars().filter(|c| !c.is_whitespace()).collect::<String>()
        );
    }
}
