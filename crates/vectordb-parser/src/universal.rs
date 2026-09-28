//! Text source detection and limited PDF text extraction.

use crate::parser::ParsedFile;
use std::collections::HashMap;
use std::path::Path;
use vectordb_core::{Chunk, Edge, EdgeKind, Node, NodeKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileCategory {
    Pdf,
    TextOrCode,
    Unsupported,
}

pub struct UniversalFileAdapter;

fn find_bytes(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    haystack
        .get(from..)?
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|offset| from + offset)
}

fn find_pdf_operator(bytes: &[u8], operator: &[u8], from: usize) -> Option<usize> {
    let mut position = from;
    while let Some(found) = find_bytes(bytes, operator, position) {
        let before = found == 0 || bytes[found - 1].is_ascii_whitespace();
        let after = found + operator.len() == bytes.len()
            || bytes[found + operator.len()].is_ascii_whitespace();
        if before && after {
            return Some(found);
        }
        position = found + operator.len();
    }
    None
}

fn pdf_literal_strings(text_object: &[u8]) -> Vec<String> {
    let mut result = Vec::new();
    let mut array_strings = Vec::new();
    let mut in_array = false;
    let mut i = 0;
    while i < text_object.len() {
        match text_object[i] {
            b'[' => {
                in_array = true;
                array_strings.clear();
                i += 1;
            }
            b']' if in_array => {
                let mut next = i + 1;
                while next < text_object.len() && text_object[next].is_ascii_whitespace() {
                    next += 1;
                }
                if text_object.get(next..next + 2) == Some(b"TJ") {
                    result.append(&mut array_strings);
                }
                in_array = false;
                i += 1;
            }
            b'(' => {
                let mut depth = 1;
                let mut next = i + 1;
                let mut literal = Vec::new();
                while next < text_object.len() && depth > 0 {
                    let byte = text_object[next];
                    if byte == b'\\' && next + 1 < text_object.len() {
                        literal.push(text_object[next + 1]);
                        next += 2;
                        continue;
                    }
                    if byte == b'(' {
                        depth += 1;
                    } else if byte == b')' {
                        depth -= 1;
                    }
                    if depth > 0 {
                        literal.push(byte);
                    }
                    next += 1;
                }
                if depth == 0 {
                    if let Ok(text) = std::str::from_utf8(&literal) {
                        let text = text.trim();
                        if text.len() >= 3 && text.chars().any(char::is_alphabetic) {
                            if in_array {
                                array_strings.push(text.to_string());
                            } else {
                                let mut operator = next;
                                while operator < text_object.len()
                                    && text_object[operator].is_ascii_whitespace()
                                {
                                    operator += 1;
                                }
                                if text_object.get(operator..operator + 2) == Some(b"Tj")
                                    || text_object.get(operator) == Some(&b'\'')
                                {
                                    result.push(text.to_string());
                                }
                            }
                        }
                    }
                }
                i = next;
            }
            _ => i += 1,
        }
    }
    result
}

impl UniversalFileAdapter {
    pub fn is_excluded_extension(ext: &str) -> bool {
        matches!(
            ext,
            "lock"
                | "tmp"
                | "log"
                | "mp3"
                | "m4a"
                | "aac"
                | "ogg"
                | "wav"
                | "flac"
                | "png"
                | "jpg"
                | "jpeg"
                | "gif"
                | "webp"
                | "svg"
                | "zip"
                | "tar"
                | "gz"
                | "7z"
                | "rar"
                | "bin"
                | "dat"
                | "exe"
                | "dll"
                | "so"
                | "dylib"
                | "wasm"
                | "docx"
                | "pptx"
                | "xlsx"
        )
    }

    /// Reject known non-text containers even when they contain valid UTF-8 bytes.
    pub fn detect_format(bytes: &[u8], path: &Path) -> FileCategory {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext == "pdf" || bytes.starts_with(b"%PDF") {
            return if bytes.starts_with(b"%PDF") {
                FileCategory::Pdf
            } else {
                FileCategory::Unsupported
            };
        }
        if Self::is_excluded_extension(&ext)
            || bytes.starts_with(b"ID3")
            || bytes.starts_with(b"\x89PNG\r\n\x1a\n")
            || bytes.starts_with(b"\xff\xd8\xff")
            || bytes.starts_with(b"PK\x03\x04")
            || bytes.starts_with(b"RIFF")
            || bytes.starts_with(b"fLaC")
            || bytes.starts_with(b"\x7fELF")
        {
            return FileCategory::Unsupported;
        }
        if std::str::from_utf8(bytes).is_ok() && !bytes.contains(&0) {
            FileCategory::TextOrCode
        } else {
            FileCategory::Unsupported
        }
    }

    /// Parse PDF document text and structure
    pub fn parse_pdf(bytes: &[u8], rel_path: &str) -> ParsedFile {
        let file_node_id = format!("pdf:{}", rel_path);
        let mut extracted_texts = Vec::new();
        let mut current_buf = Vec::new();
        let mut cursor = 0;
        while let Some(stream_start) = find_bytes(bytes, b"stream", cursor) {
            let mut data_start = stream_start + b"stream".len();
            if bytes.get(data_start) == Some(&b'\r') {
                data_start += 1;
            }
            if bytes.get(data_start) == Some(&b'\n') {
                data_start += 1;
            }
            let Some(stream_end) = find_bytes(bytes, b"endstream", data_start) else {
                break;
            };
            let stream = &bytes[data_start..stream_end];
            let mut text_cursor = 0;
            while let Some(begin) = find_pdf_operator(stream, b"BT", text_cursor) {
                let Some(end) = find_pdf_operator(stream, b"ET", begin + 2) else {
                    break;
                };
                for text in pdf_literal_strings(&stream[begin + 2..end]) {
                    current_buf.push(text);
                    let current_chars: usize = current_buf.iter().map(String::len).sum();
                    if current_chars >= 800 {
                        extracted_texts.push(current_buf.join(" "));
                        current_buf.clear();
                    }
                }
                text_cursor = end + 2;
            }
            cursor = stream_end + b"endstream".len();
            if extracted_texts.len() >= 50 {
                break;
            }
        }

        if !current_buf.is_empty() && extracted_texts.len() < 50 {
            extracted_texts.push(current_buf.join(" "));
        }

        let doc_title = Path::new(rel_path)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let doc_node = Node {
            id: file_node_id.clone(),
            label: doc_title.clone(),
            kind: NodeKind::Document,
            file_path: rel_path.to_string(),
            start_line: 1,
            end_line: extracted_texts.len(),
            signature: Some(format!("Document: {}", doc_title)),
            docstring: Some(format!("Pages/Sections: {}", extracted_texts.len())),
            chunk_id: None,
            vector_id: None,
            metadata: HashMap::new(),
        };

        let mut chunks = Vec::new();
        let mut nodes = vec![doc_node];
        let mut edges = Vec::new();

        for (idx, text) in extracted_texts.into_iter().enumerate() {
            let section_idx = idx + 1;
            let chunk_id = format!("chunk:{}:sec{}", rel_path, section_idx);
            let sec_node_id = format!("doc:{}:sec{}", rel_path, section_idx);

            let sec_node = Node {
                id: sec_node_id.clone(),
                label: format!("{} (Sec {})", doc_title, section_idx),
                kind: NodeKind::DocSection,
                file_path: rel_path.to_string(),
                start_line: section_idx,
                end_line: section_idx,
                signature: None,
                docstring: None,
                chunk_id: Some(chunk_id.clone()),
                vector_id: None,
                metadata: HashMap::new(),
            };

            let chunk = Chunk {
                id: chunk_id.clone(),
                file_path: rel_path.to_string(),
                start_line: section_idx,
                end_line: section_idx,
                tokens_approx: text.split_whitespace().count(),
                content: text,
                summary: None,
                node_id: Some(sec_node_id.clone()),
                vector_id: 0,
            };

            edges.push(Edge {
                source: file_node_id.clone(),
                target: sec_node_id,
                kind: EdgeKind::Contains,
                weight: 1.0,
            });

            nodes.push(sec_node);
            chunks.push(chunk);
        }

        ParsedFile {
            file_path: rel_path.to_string(),
            chunks,
            nodes,
            edges,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_only_text_showing_pdf_literals() {
        let source = b"%PDF-1.4\n/Title (metadata only)\nstream\nBT (Visible heading) Tj [(Split) -20 (paragraph)] TJ ET\nendstream";
        let parsed = UniversalFileAdapter::parse_pdf(source, "paper.pdf");
        assert_eq!(parsed.chunks.len(), 1);
        assert_eq!(parsed.chunks[0].content, "Visible heading Split paragraph");
    }
}
