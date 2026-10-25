# third_party

Third-party source code only. Install scripts are in `script/setup/`.

## CUTLASS (pending)

`.gitmodules` lists `third_party/cutlass`, but the repo has no submodule commit for it yet.
To add it, pinned to `v4.2.1` (the tag in `script/setup/install_cutlass.sh`):

```bash
git submodule add --force https://github.com/NVIDIA/cutlass.git third_party/cutlass
git -C third_party/cutlass checkout v4.2.1
git add .gitmodules third_party/cutlass
```

`--force` reuses the clone in `.git/modules/third_party/cutlass` (no download) if it exists.
Use CUTLASS as header-only: `third_party/cutlass/include` is on the include path when it exists.
