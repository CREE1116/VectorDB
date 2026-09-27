//! Universal File Adapter: Multi-modal metadata, text extraction,
//! and binary byte-fingerprinting for ANY file format (Audio/MP3, PDF, Images, Binaries).

use crate::parser::ParsedFile;
use std::collections::HashMap;
use std::path::Path;
use vectordb_core::{Chunk, Edge, EdgeKind, Node, NodeKind};

#[derive(Debug, Clone, PartialEq)]
pub enum FileCategory {
    Audio(String),
    Document(String),
    Image(String),
    Archive(String),
    Binary(String),
    TextOrCode(String),
}

/// Standard ID3v1 Genre code mapping
fn get_id3_genre(code: u8) -> &'static str {
    match code {
        0 => "Blues",
        1 => "Classic Rock",
        2 => "Country",
        3 => "Dance",
        4 => "Disco",
        5 => "Funk",
        6 => "Grunge",
        7 => "Hip-Hop",
        8 => "Jazz",
        9 => "Metal",
        10 => "New Age",
        11 => "Oldies",
        12 => "Other",
        13 => "Pop",
        14 => "R&B",
        15 => "Rap",
        16 => "Reggae",
        17 => "Rock",
        18 => "Techno",
        19 => "Industrial",
        20 => "Alternative",
        21 => "Ska",
        22 => "Death Metal",
        23 => "Pranks",
        24 => "Soundtrack",
        25 => "Euro-Techno",
        26 => "Ambient",
        27 => "Trip-Hop",
        28 => "Vocal",
        29 => "Jazz+Funk",
        30 => "Fusion",
        31 => "Trance",
        32 => "Classical",
        33 => "Instrumental",
        34 => "Acid",
        35 => "House",
        36 => "Game",
        37 => "Sound Clip",
        38 => "Gospel",
        39 => "Noise",
        40 => "Alternative Rock",
        41 => "Bass",
        42 => "Soul",
        43 => "Punk",
        44 => "Space",
        45 => "Meditative",
        46 => "Instrumental Pop",
        47 => "Instrumental Rock",
        48 => "Ethnic",
        49 => "Gothic",
        50 => "Darkwave",
        51 => "Techno-Industrial",
        52 => "Electronic",
        53 => "Pop-Folk",
        54 => "Eurodance",
        55 => "Dream",
        56 => "Southern Rock",
        57 => "Comedy",
        58 => "Cult",
        59 => "Gangsta",
        60 => "Top 40",
        61 => "Christian Rap",
        62 => "Pop/Funk",
        63 => "Jungle",
        64 => "Native US",
        65 => "Cabaret",
        66 => "New Wave",
        67 => "Psychadelic",
        68 => "Rave",
        69 => "Showtunes",
        70 => "Trailer",
        71 => "Lo-Fi",
        72 => "Tribal",
        73 => "Acid Punk",
        74 => "Acid Jazz",
        75 => "Polka",
        76 => "Retro",
        77 => "Musical",
        78 => "Rock & Roll",
        79 => "Hard Rock",
        80 => "Folk",
        81 => "Folk-Rock",
        82 => "National Folk",
        83 => "Swing",
        84 => "Fast Fusion",
        85 => "Bebob",
        86 => "Latin",
        87 => "Revival",
        88 => "Celtic",
        89 => "Bluegrass",
        90 => "Avantgarde",
        91 => "Gothic Rock",
        92 => "Progressive Rock",
        93 => "Psychedelic Rock",
        94 => "Symphonic Rock",
        95 => "Slow Rock",
        96 => "Big Band",
        97 => "Chorus",
        98 => "Easy Listening",
        99 => "Acoustic",
        100 => "Humour",
        101 => "Speech",
        102 => "Chanson",
        103 => "Opera",
        104 => "Chamber Music",
        105 => "Sonata",
        106 => "Symphony",
        107 => "Booty Bass",
        108 => "Primus",
        109 => "Porn Groove",
        110 => "Satire",
        111 => "Slow Jam",
        112 => "Club",
        113 => "Tango",
        114 => "Samba",
        115 => "Folklore",
        116 => "Ballad",
        117 => "Power Ballad",
        118 => "Rhythmic Soul",
        119 => "Freestyle",
        120 => "Duet",
        121 => "Punk Rock",
        122 => "Drum Solo",
        123 => "A capella",
        124 => "Euro-House",
        125 => "Dance Hall",
        _ => "Unknown Genre",
    }
}

pub struct UniversalFileAdapter;

impl UniversalFileAdapter {
    /// Detect format from magic bytes and file extension
    pub fn detect_format(bytes: &[u8], path: &Path) -> FileCategory {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // 1. Magic bytes detection
        if bytes.len() >= 3 && &bytes[0..3] == b"ID3" {
            return FileCategory::Audio("audio/mp3".into());
        }
        if bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0 {
            return FileCategory::Audio("audio/mp3".into());
        }
        if bytes.len() >= 4 && &bytes[0..4] == b"%PDF" {
            return FileCategory::Document("application/pdf".into());
        }
        if bytes.len() >= 8 && &bytes[0..8] == b"\x89PNG\r\n\x1a\n" {
            return FileCategory::Image("image/png".into());
        }
        if bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
            return FileCategory::Image("image/jpeg".into());
        }
        if bytes.len() >= 4
            && &bytes[0..4] == b"RIFF"
            && bytes.len() >= 12
            && &bytes[8..12] == b"WAVE"
        {
            return FileCategory::Audio("audio/wav".into());
        }
        if bytes.len() >= 4 && &bytes[0..4] == b"fLaC" {
            return FileCategory::Audio("audio/flac".into());
        }
        if bytes.len() >= 4 && &bytes[0..4] == b"PK\x03\x04" {
            if ext == "docx" || ext == "pptx" || ext == "xlsx" {
                return FileCategory::Document(format!("application/vnd.openxmlformats-{}", ext));
            }
            return FileCategory::Archive("application/zip".into());
        }

        // 2. Extension fallbacks
        match ext.as_str() {
            "mp3" | "m4a" | "aac" | "ogg" | "wav" | "flac" => {
                FileCategory::Audio(format!("audio/{}", ext))
            }
            "pdf" => FileCategory::Document("application/pdf".into()),
            "png" | "jpg" | "jpeg" | "gif" | "webp" | "svg" => {
                FileCategory::Image(format!("image/{}", ext))
            }
            "zip" | "tar" | "gz" | "7z" | "rar" => {
                FileCategory::Archive(format!("archive/{}", ext))
            }
            "bin" | "dat" | "exe" | "dll" | "so" | "dylib" | "wasm" => {
                FileCategory::Binary("application/octet-stream".into())
            }
            _ => {
                // Check if UTF-8 text
                if std::str::from_utf8(bytes).is_ok() {
                    FileCategory::TextOrCode("text/plain".into())
                } else {
                    FileCategory::Binary("application/octet-stream".into())
                }
            }
        }
    }

    /// Parse MP3 / Audio metadata (ID3v1 and ID3v2) and build rich Knowledge Graph elements
    pub fn parse_audio(bytes: &[u8], rel_path: &str) -> (ParsedFile, Option<Vec<f32>>) {
        let mut title = Path::new(rel_path)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let mut artist = "Unknown Artist".to_string();
        let mut album = "Unknown Album".to_string();
        let mut year = "Unknown Year".to_string();
        let mut genre = "Unknown Genre".to_string();

        // Check ID3v2 at the beginning
        if bytes.len() >= 10 && &bytes[0..3] == b"ID3" {
            let mut pos = 10;
            while pos + 10 < bytes.len() && pos < 4096 {
                let frame_id = match std::str::from_utf8(&bytes[pos..pos + 4]) {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let size = ((bytes[pos + 4] as usize) << 24)
                    | ((bytes[pos + 5] as usize) << 16)
                    | ((bytes[pos + 6] as usize) << 8)
                    | (bytes[pos + 7] as usize);

                if size == 0 || pos + 10 + size > bytes.len() {
                    break;
                }

                let frame_content = &bytes[pos + 10..pos + 10 + size];
                let text = String::from_utf8_lossy(frame_content)
                    .trim_matches('\0')
                    .trim()
                    .to_string();

                match frame_id {
                    "TIT2" => {
                        if !text.is_empty() {
                            title = text;
                        }
                    }
                    "TPE1" => {
                        if !text.is_empty() {
                            artist = text;
                        }
                    }
                    "TALB" => {
                        if !text.is_empty() {
                            album = text;
                        }
                    }
                    "TCON" => {
                        if !text.is_empty() {
                            genre = text;
                        }
                    }
                    "TYER" | "TDRC" if !text.is_empty() => {
                        year = text;
                    }
                    _ => {}
                }
                pos += 10 + size;
            }
        }

        // Check ID3v1 at the end (last 128 bytes)
        if bytes.len() >= 128 {
            let tag_slice = &bytes[bytes.len() - 128..];
            if &tag_slice[0..3] == b"TAG" {
                let v1_title = String::from_utf8_lossy(&tag_slice[3..33])
                    .trim_matches('\0')
                    .trim()
                    .to_string();
                let v1_artist = String::from_utf8_lossy(&tag_slice[33..63])
                    .trim_matches('\0')
                    .trim()
                    .to_string();
                let v1_album = String::from_utf8_lossy(&tag_slice[63..93])
                    .trim_matches('\0')
                    .trim()
                    .to_string();
                let v1_year = String::from_utf8_lossy(&tag_slice[93..97])
                    .trim_matches('\0')
                    .trim()
                    .to_string();
                let genre_code = tag_slice[127];

                if !v1_title.is_empty() && title.contains('.') {
                    title = v1_title;
                }
                if !v1_artist.is_empty() && artist == "Unknown Artist" {
                    artist = v1_artist;
                }
                if !v1_album.is_empty() && album == "Unknown Album" {
                    album = v1_album;
                }
                if !v1_year.is_empty() && year == "Unknown Year" {
                    year = v1_year;
                }
                if genre == "Unknown Genre" {
                    genre = get_id3_genre(genre_code).to_string();
                }
            }
        }

        let file_node_id = format!("audio:{}", rel_path);
        let artist_node_id = format!("artist:{}", artist);
        let album_node_id = format!("album:{}", album);
        let chunk_id = format!("chunk:{}:meta", rel_path);

        let content_text = format!(
            "Audio Track: {}\nArtist: {}\nAlbum: {}\nGenre: {}\nYear: {}\nFile Path: {}",
            title, artist, album, genre, year, rel_path
        );

        let track_node = Node {
            id: file_node_id.clone(),
            label: format!("{} - {}", artist, title),
            kind: NodeKind::AudioTrack,
            file_path: rel_path.to_string(),
            start_line: 1,
            end_line: 1,
            signature: Some(format!("Audio: {} [{}]", title, genre)),
            docstring: Some(format!("Album: {}, Year: {}", album, year)),
            chunk_id: Some(chunk_id.clone()),
            vector_id: None,
            metadata: HashMap::from([
                ("title".into(), title.clone()),
                ("artist".into(), artist.clone()),
                ("album".into(), album.clone()),
                ("genre".into(), genre.clone()),
                ("year".into(), year.clone()),
            ]),
        };

        let artist_node = Node {
            id: artist_node_id.clone(),
            label: artist.clone(),
            kind: NodeKind::Concept,
            file_path: rel_path.to_string(),
            start_line: 1,
            end_line: 1,
            signature: Some(format!("Artist: {}", artist)),
            docstring: None,
            chunk_id: None,
            vector_id: None,
            metadata: HashMap::new(),
        };

        let album_node = Node {
            id: album_node_id.clone(),
            label: album.clone(),
            kind: NodeKind::Module,
            file_path: rel_path.to_string(),
            start_line: 1,
            end_line: 1,
            signature: Some(format!("Album: {}", album)),
            docstring: None,
            chunk_id: None,
            vector_id: None,
            metadata: HashMap::new(),
        };

        let chunk = Chunk {
            id: chunk_id,
            file_path: rel_path.to_string(),
            start_line: 1,
            end_line: 1,
            tokens_approx: content_text.split_whitespace().count(),
            content: content_text,
            summary: Some(format!("{} by {}", title, artist)),
            node_id: Some(file_node_id.clone()),
            vector_id: 0,
        };

        let edges = vec![
            Edge {
                source: file_node_id.clone(),
                target: artist_node_id,
                kind: EdgeKind::CreatedBy,
                weight: 1.0,
            },
            Edge {
                source: file_node_id,
                target: album_node_id,
                kind: EdgeKind::BelongsTo,
                weight: 1.0,
            },
        ];

        let parsed = ParsedFile {
            file_path: rel_path.to_string(),
            chunks: vec![chunk],
            nodes: vec![track_node, artist_node, album_node],
            edges,
            custom_vector: None,
        };

        (parsed, None)
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

        // If no literal strings extracted (e.g. encoded), create document descriptor
        if extracted_texts.is_empty() {
            extracted_texts.push(format!(
                "PDF Document: {}\nSize: {} bytes\nFormat: Portable Document Format (PDF)",
                rel_path,
                bytes.len()
            ));
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
            custom_vector: None,
        }
    }

    /// Parse ANY binary or unknown file using Byte-Frequency Histogram (256-dim) & Shannon Entropy
    pub fn parse_binary_fingerprint(
        bytes: &[u8],
        rel_path: &str,
        format_desc: &str,
    ) -> (ParsedFile, Vec<f32>) {
        let file_node_id = format!("bin:{}", rel_path);
        let chunk_id = format!("chunk:{}:fp", rel_path);

        // 1. Calculate 256-bin Byte Frequency Histogram
        let mut hist = vec![0.0f32; 256];
        if !bytes.is_empty() {
            for &b in bytes {
                hist[b as usize] += 1.0;
            }
            // Normalize histogram to unit vector (L2 norm = 1.0)
            vectordb_core::simd::normalize_in_place(&mut hist);
        }

        // 2. Calculate Shannon Entropy (bits per byte: 0.0 ~ 8.0)
        let total_bytes = bytes.len() as f64;
        let mut entropy = 0.0f64;
        if total_bytes > 0.0 {
            let mut counts = [0usize; 256];
            for &b in bytes {
                counts[b as usize] += 1;
            }
            for &c in &counts {
                if c > 0 {
                    let p = (c as f64) / total_bytes;
                    entropy -= p * p.log2();
                }
            }
        }

        let entropy_desc = if entropy > 7.5 {
            "High (Compressed / Encrypted)"
        } else if entropy > 5.0 {
            "Medium (Compiled Code / Structured Data)"
        } else {
            "Low (Plain Text / Repetitive Data)"
        };

        let file_name = Path::new(rel_path)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let content_text = format!(
            "Binary File: {}\nFormat: {}\nSize: {} bytes\nEntropy: {:.2} bits/byte ({})\nFingerprint: 256-dim byte frequency distribution",
            file_name, format_desc, bytes.len(), entropy, entropy_desc
        );

        let bin_node = Node {
            id: file_node_id.clone(),
            label: file_name.clone(),
            kind: NodeKind::Binary,
            file_path: rel_path.to_string(),
            start_line: 1,
            end_line: 1,
            signature: Some(format!("{} ({} bytes)", format_desc, bytes.len())),
            docstring: Some(format!("Entropy: {:.2} ({})", entropy, entropy_desc)),
            chunk_id: Some(chunk_id.clone()),
            vector_id: None,
            metadata: HashMap::from([
                ("format".into(), format_desc.to_string()),
                ("size".into(), bytes.len().to_string()),
                ("entropy".into(), format!("{:.2}", entropy)),
            ]),
        };

        let chunk = Chunk {
            id: chunk_id,
            file_path: rel_path.to_string(),
            start_line: 1,
            end_line: 1,
            tokens_approx: content_text.split_whitespace().count(),
            content: content_text,
            summary: Some(format!("{} ({})", file_name, format_desc)),
            node_id: Some(file_node_id),
            vector_id: 0,
        };

        let parsed = ParsedFile {
            file_path: rel_path.to_string(),
            chunks: vec![chunk],
            nodes: vec![bin_node],
            edges: Vec::new(),
            custom_vector: Some(hist.clone()),
        };

        // Return parsed structure and the 256-dim byte frequency vector
        (parsed, hist)
    }
}
