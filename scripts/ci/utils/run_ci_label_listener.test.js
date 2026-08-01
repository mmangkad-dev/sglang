const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const {
  ALLOWED_WORKFLOW_PATHS,
  isAllowedRunForPull,
} = require("./run_ci_label_listener");

const pullRequest = {
  number: 33220,
  head: {
    sha: "abc123",
    ref: "contributor/run-ci-fix",
    repo: { full_name: "external-contributor/sglang" },
  },
};

const sameRepoPullRequest = {
  number: 33219,
  head: {
    sha: "same123",
    ref: "maintainer/run-ci-fix",
    repo: { full_name: "sgl-project/sglang" },
  },
};

function run(overrides = {}) {
  return {
    path: ".github/workflows/pr-test.yml",
    head_sha: "abc123",
    head_branch: "contributor/run-ci-fix",
    head_repository: { full_name: "external-contributor/sglang" },
    pull_requests: [],
    ...overrides,
  };
}

test("accepts a same-repository run explicitly associated with the PR", () => {
  assert.equal(
    isAllowedRunForPull(
      run({
        head_sha: "same123",
        head_branch: "maintainer/run-ci-fix",
        head_repository: { full_name: "sgl-project/sglang" },
        pull_requests: [{ number: 33219 }],
      }),
      sameRepoPullRequest,
    ),
    true,
  );
});

test("accepts a fork run with an empty association array by exact head identity", () => {
  assert.equal(isAllowedRunForPull(run(), pullRequest), true);
});

test("rejects an explicit association with a different PR", () => {
  assert.equal(
    isAllowedRunForPull(run({ pull_requests: [{ number: 33221 }] }), pullRequest),
    false,
  );
});

test("rejects incomplete fork identity matches", () => {
  assert.equal(
    isAllowedRunForPull(run({ head_branch: "another-branch" }), pullRequest),
    false,
  );
  assert.equal(
    isAllowedRunForPull(
      run({ head_repository: { full_name: "someone-else/sglang" } }),
      pullRequest,
    ),
    false,
  );
  assert.equal(isAllowedRunForPull(run({ head_sha: "def456" }), pullRequest), false);
});

test("rejects workflows outside the live pr-gate allowlist", () => {
  assert.equal(
    isAllowedRunForPull(
      run({ path: ".github/workflows/pr-test-rust.yml" }),
      pullRequest,
    ),
    false,
  );
  assert.equal(
    isAllowedRunForPull(run({ path: ".github/workflows/lint.yml" }), pullRequest),
    false,
  );
});

test("every allowlisted workflow uses the live gate and does not handle labels", () => {
  const repositoryRoot = path.resolve(__dirname, "../../..");
  for (const workflow of ALLOWED_WORKFLOW_PATHS) {
    const contents = fs.readFileSync(path.join(repositoryRoot, workflow), "utf8");
    assert.match(contents, /uses: \.\/\.github\/workflows\/pr-gate\.yml/);
    assert.doesNotMatch(contents, /types:.*\blabeled\b/);
  }
});
