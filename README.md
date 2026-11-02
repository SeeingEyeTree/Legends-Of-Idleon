# Legends of Idleon helpers

A collection of small quality-of-life automations for the game **Legends of Idleon**.
It drives the game by simulating mouse/keyboard input and reading the screen with
OpenCV, so it is **Windows only**.

## What's in it

| Command | What it does |
| --- | --- |
| `cook <ladles>` | Withdraws ladles from the chest, fills every kitchen with meals and levels them up. |
| `anvil` | Collects finished items from all anvils. |
| `vials <loops>` | Repeats the vial routine. Start with your inventory open. |
| `read-worship charge-rate\|cur-charge` | Reads a number from the Worship screen using digit templates. |
| `tree-bot` | Experimental, unfinished timing bot. |

Also included as functions (not yet wired to the CLI): shop buying (`tasks/shops.py`) and the World 2
boss attack-order macro (`tasks/boss.py`).

## Setup

You need Python 3.9 or newer.

```
git clone https://github.com/SeeingEyeTree/Legends-Of-Idleon.git
cd Legends-Of-Idleon
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Usage

Run a command, then switch to the game window during the 5-second countdown:

```
python -m legends_of_idleon cook 3
python -m legends_of_idleon anvil
python -m legends_of_idleon vials 10
```

(`idleon cook 3` also works after `pip install -e .`.) Use `--no-countdown` to skip the wait.

## Automation stack (screenshot-driven)

`legends_of_idleon/stack/` runs game flows as verified steps: act, look at the screen, confirm the
expected template appeared, retry if not. Templates live in `legends_of_idleon/templates/ui/`.

Tools for building templates:

```
python -m legends_of_idleon stack snap shot.png                      # 1920x1080 screenshot
python -m legends_of_idleon stack crop shot.png play_button 800 600 1100 680
python -m legends_of_idleon stack find shot.png play_button          # score a template
```

### Remote testing agent

To drive the game from another machine (or a Claude session) without giving it a shell, run the
agent on the game PC and share a folder:

```
python -m legends_of_idleon agent C:\dev\Legends-Of-Idleon\bridge
```

It only understands `ping`, `snap`, `click`, `move`, `key`, `find` and `sleep` jobs dropped in
`bridge/jobs/*.json`; results and screenshots appear in `bridge/results/` and `bridge/out/`.
Stop it with Ctrl+C.

## Screen resolution

Click coordinates were recorded on a **2560x1600** display and are scaled to your screen size, but
they assume the game fills the screen in the same layout. The digit images in
`legends_of_idleon/templates/inventory_digits` were captured at **1920x1080**; on other resolutions
screenshot your own digits and replace them.

## Known gaps

- `find_player()` (used by `anvil` and `boss.active_collect`) is not implemented yet.
- `tree-bot` is an unfinished prototype.

## Layout

```
legends_of_idleon/
  keys.py        scan-code keyboard input (SendInput)
  controls.py    scaled mouse clicks, key presses, inventory slot picking
  screen.py      fast screen capture
  vision.py      template matching and colour helpers
  digits.py      reads numbers from the UI
  hotkeys.py     polls held keys
  cli.py         command-line entry point
  tasks/         cooking, anvil, vials, shops, boss, tree_bot
  templates/     reference images used for matching
tests/           tests for the platform-independent helpers
```

## Development

```
pip install -e .[dev]
ruff check .
pytest
```
