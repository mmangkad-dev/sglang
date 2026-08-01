const ALLOWED_WORKFLOW_PATHS = new Set([
  ".github/workflows/pr-test.yml",
  ".github/workflows/pr-test-amd.yml",
  ".github/workflows/pr-test-arm64.yml",
  ".github/workflows/pr-test-mlx.yml",
  ".github/workflows/pr-test-musa.yml",
  ".github/workflows/pr-test-npu.yml",
  ".github/workflows/pr-test-xeon.yml",
  ".github/workflows/pr-test-xpu.yml",
]);

function workflowPath(run) {
  // GitHub normally returns `.github/workflows/foo.yml`; tolerate the
  // `path@ref` form used by some API responses without widening the allowlist.
  return (run.path || "").split("@")[0];
}

function isAllowedRunForPull(run, pullRequest) {
  if (!ALLOWED_WORKFLOW_PATHS.has(workflowPath(run))) return false;
  if (run.head_sha !== pullRequest.head.sha) return false;

  const associatedPulls = run.pull_requests || [];
  if (associatedPulls.length > 0) {
    return associatedPulls.some((pull) => pull.number === pullRequest.number);
  }

  // GitHub returns pull_requests: [] for workflow runs from fork PRs. Match
  // all three immutable/head identity fields before accepting that fallback.
  return (
    run.head_repository?.full_name === pullRequest.head.repo.full_name &&
    run.head_branch === pullRequest.head.ref &&
    run.head_sha === pullRequest.head.sha
  );
}

module.exports = { ALLOWED_WORKFLOW_PATHS, isAllowedRunForPull };
