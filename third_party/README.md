# third_party

Third-party source code only. Install scripts are in `script/setup/`.

## CUTLASS

`third_party/cutlass` is a git submodule, pinned to `v4.8.1` (the tag in `script/setup/install_cutlass.sh`).
The include path of the repo is the install in `libs/cutlass/include`, not this source tree: that one tree
holds `cutlass/`, `cute/` and `cutlass/util/` together. Run `script/setup/install_cutlass.sh` once after a clone.

Get it after a clone:

```bash
git submodule update --init third_party/cutlass
```

Move it to another tag (example: `v4.8.1`). Fetch the tag first: the local clone has only the tags it fetched before.

```bash
git -C third_party/cutlass fetch origin tag v4.8.1 --no-tags
git -C third_party/cutlass checkout v4.8.1
git add third_party/cutlass
```

Then set the same tag in `script/setup/install_cutlass.sh`.
