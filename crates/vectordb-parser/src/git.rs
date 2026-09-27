//! Git commit history analyzer for evolutionary coupling & co-change discovery.
//! Identifies files and modules that frequently change together across commits.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitRecord {
    pub hash: String,
    pub subject: String,
    pub files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoChangeCandidate {
    pub file_path: String,
    pub co_change_count: usize,
    pub total_commits: usize,
    pub confidence: f32,
    pub recent_commits: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitCoChangeReport {
    pub target_file: String,
    pub target_commit_count: usize,
    pub total_commits_analyzed: usize,
    pub candidates: Vec<CoChangeCandidate>,
}

pub struct GitCoChangeAnalyzer;

impl GitCoChangeAnalyzer {
    /// Extract commit history from the git repository using `git log`.
    /// Runs in ~10-20ms without any background daemons.
    pub fn extract_history(
        repo_root: &Path,
        max_commits: usize,
    ) -> anyhow::Result<Vec<CommitRecord>> {
        let output = match Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args([
                "log",
                "--name-only",
                &format!("-n{}", max_commits),
                "--pretty=format:COMMIT:%h|%s",
            ])
            .output()
        {
            Ok(o) => o,
            Err(_) => return Ok(Vec::new()),
        };

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut commits = Vec::new();
        let mut cur_hash = String::new();
        let mut cur_subject = String::new();
        let mut cur_files = Vec::new();

        for line in stdout.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some(meta) = line.strip_prefix("COMMIT:") {
                if !cur_hash.is_empty() && !cur_files.is_empty() {
                    commits.push(CommitRecord {
                        hash: cur_hash.clone(),
                        subject: cur_subject.clone(),
                        files: cur_files.clone(),
                    });
                    cur_files.clear();
                }
                if let Some((h, s)) = meta.split_once('|') {
                    cur_hash = h.to_string();
                    cur_subject = s.to_string();
                } else {
                    cur_hash = meta.to_string();
                    cur_subject = String::new();
                }
            } else {
                cur_files.push(line.to_string());
            }
        }

        if !cur_hash.is_empty() && !cur_files.is_empty() {
            commits.push(CommitRecord {
                hash: cur_hash,
                subject: cur_subject,
                files: cur_files,
            });
        }

        Ok(commits)
    }

    /// Analyze co-change coupling for a specific file across git commits.
    pub fn analyze_file(commits: &[CommitRecord], target_file: &str) -> GitCoChangeReport {
        let target_norm = target_file.trim_start_matches("./");
        let mut target_commit_count = 0;
        let mut co_occurrence: HashMap<String, (usize, Vec<String>)> = HashMap::new();
        let mut file_commit_counts: HashMap<String, usize> = HashMap::new();

        for commit in commits {
            let files_set: HashSet<&str> = commit
                .files
                .iter()
                .map(|f| f.trim_start_matches("./"))
                .collect();
            for f in &files_set {
                *file_commit_counts.entry(f.to_string()).or_insert(0) += 1;
            }

            let matches_target = files_set
                .iter()
                .any(|f| *f == target_norm || f.ends_with(target_norm));
            if matches_target {
                target_commit_count += 1;
                for f in &files_set {
                    if *f != target_norm && !f.ends_with(target_norm) {
                        let entry = co_occurrence
                            .entry(f.to_string())
                            .or_insert((0, Vec::new()));
                        entry.0 += 1;
                        if entry.1.len() < 3 {
                            entry
                                .1
                                .push(format!("{} - {}", commit.hash, commit.subject));
                        }
                    }
                }
            }
        }

        let mut candidates = Vec::new();
        for (file_path, (count, recent_commits)) in co_occurrence {
            let confidence = if target_commit_count > 0 {
                count as f32 / target_commit_count as f32
            } else {
                0.0
            };
            let total = *file_commit_counts.get(&file_path).unwrap_or(&count);
            candidates.push(CoChangeCandidate {
                file_path,
                co_change_count: count,
                total_commits: total,
                confidence,
                recent_commits,
            });
        }

        candidates.sort_by(|a, b| {
            b.co_change_count.cmp(&a.co_change_count).then(
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        GitCoChangeReport {
            target_file: target_norm.to_string(),
            target_commit_count,
            total_commits_analyzed: commits.len(),
            candidates,
        }
    }

    /// Build co-change edges for the Knowledge Graph across all files in the repository.
    pub fn build_co_change_edges(
        commits: &[CommitRecord],
        min_co_occurrences: usize,
    ) -> Vec<(String, String, f32)> {
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        let mut file_counts: HashMap<String, usize> = HashMap::new();

        for commit in commits {
            let files: Vec<String> = commit
                .files
                .iter()
                .map(|f| f.trim_start_matches("./").to_string())
                .collect();
            for f in &files {
                *file_counts.entry(f.clone()).or_insert(0) += 1;
            }

            for i in 0..files.len() {
                for j in (i + 1)..files.len() {
                    let a = &files[i];
                    let b = &files[j];
                    let key = if a < b {
                        (a.clone(), b.clone())
                    } else {
                        (b.clone(), a.clone())
                    };
                    *pair_counts.entry(key).or_insert(0) += 1;
                }
            }
        }

        let mut edges = Vec::new();
        for ((a, b), count) in pair_counts {
            if count >= min_co_occurrences {
                let count_a = *file_counts.get(&a).unwrap_or(&count);
                let count_b = *file_counts.get(&b).unwrap_or(&count);
                let union = count_a + count_b - count;
                let jaccard = if union > 0 {
                    count as f32 / union as f32
                } else {
                    0.0
                };
                edges.push((a, b, jaccard));
            }
        }

        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_co_change_analysis() {
        let commits = vec![
            CommitRecord {
                hash: "c1".into(),
                subject: "feat: add schema".into(),
                files: vec!["src/config.rs".into(), "src/schema.rs".into()],
            },
            CommitRecord {
                hash: "c2".into(),
                subject: "fix: update schema validation".into(),
                files: vec![
                    "src/config.rs".into(),
                    "src/schema.rs".into(),
                    "tests/schema_test.rs".into(),
                ],
            },
            CommitRecord {
                hash: "c3".into(),
                subject: "docs: update readme".into(),
                files: vec!["README.md".into()],
            },
        ];

        let report = GitCoChangeAnalyzer::analyze_file(&commits, "src/config.rs");
        assert_eq!(report.target_commit_count, 2);
        assert_eq!(report.candidates.len(), 2);
        assert_eq!(report.candidates[0].file_path, "src/schema.rs");
        assert_eq!(report.candidates[0].co_change_count, 2);
        assert!((report.candidates[0].confidence - 1.0).abs() < f32::EPSILON);

        let edges = GitCoChangeAnalyzer::build_co_change_edges(&commits, 1);
        assert!(edges
            .iter()
            .any(|(a, b, _)| (a == "src/config.rs" && b == "src/schema.rs")
                || (a == "src/schema.rs" && b == "src/config.rs")));
    }
}
