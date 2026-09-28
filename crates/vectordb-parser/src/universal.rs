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

        // 1. Resilient text extraction from PDF stream objects
        let mut i = 0;
        let mut current_buf = Vec::new();
        while i < bytes.len() {
            // Find literal strings in parenthesis `(...)`
            if bytes[i] == b'(' {
                let mut depth = 1;
                let mut j = i + 1;
                let mut text_bytes = Vec::new();
                while j < bytes.len() && depth > 0 {
                    if bytes[j] == b'\\' && j + 1 < bytes.len() {
                        text_bytes.push(bytes[j + 1]);
                        j += 2;
                        continue;
                    }
                    if bytes[j] == b'(' {
                        depth += 1;
                    } else if bytes[j] == b')' {
                        depth -= 1;
                    }

                    if depth > 0 {
                        text_bytes.push(bytes[j]);
                    }
                    j += 1;
                }

                if let Ok(s) = std::str::from_utf8(&text_bytes) {
                    let clean = s.trim();
                    if clean.len() >= 3 && clean.chars().any(|c| c.is_alphabetic()) {
                        current_buf.push(clean.to_string());
                        let current_chars: usize = current_buf.iter().map(|s| s.len()).sum();
                        if current_chars >= 800 {
                            extracted_texts.push(current_buf.join(" "));
                            current_buf.clear();
                            if extracted_texts.len() >= 50 {
                                break;
                            }
                        }
                    }
                }
                i = j;
                continue;
            }
            i += 1;
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
