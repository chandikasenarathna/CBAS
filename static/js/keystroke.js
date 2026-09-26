/**
 * keystroke.js — Browser-based keystroke dynamics capture engine.
 *
 * Attaches to a textarea/input and records precise press/release timestamps.
 * Sends event batches to the server for feature extraction and scoring.
 *
 * This replaces pynput — all capture happens in the browser via JS,
 * which means we capture ONLY what the user types in our app,
 * not system-wide keystrokes.
 */

class KeystrokeCapture {
  constructor(options = {}) {
    this.windowSize  = options.windowSize || 30;
    this.onWindow    = options.onWindow   || null;    // callback(events)
    this.element     = null;
    this._pressMap   = {};        // key -> timestamp
    this._events     = [];        // {key, press_time, release_time}
    this._active     = false;
    this._totalKeys  = 0;

    // Bind handlers
    this._onKeyDown = this._handleKeyDown.bind(this);
    this._onKeyUp   = this._handleKeyUp.bind(this);
  }

  /** Attach to a DOM element and start capturing. */
  attach(element) {
    this.element = element;
    element.addEventListener("keydown", this._onKeyDown);
    element.addEventListener("keyup",   this._onKeyUp);
    this._active = true;
  }

  /** Stop capturing and detach. */
  detach() {
    if (this.element) {
      this.element.removeEventListener("keydown", this._onKeyDown);
      this.element.removeEventListener("keyup",   this._onKeyUp);
    }
    this._active = false;
  }

  /** Get all events collected so far and clear buffer. */
  flush() {
    const evts = [...this._events];
    this._events = [];
    return evts;
  }

  /** Get total keystrokes captured this session. */
  get totalKeys() { return this._totalKeys; }

  /** Reset everything. */
  reset() {
    this._pressMap  = {};
    this._events    = [];
    this._totalKeys = 0;
  }

  // ── Internal handlers ────────────────────────────────────────────────────

  _handleKeyDown(e) {
    if (!this._active) return;

    // Skip modifier-only keys, function keys, arrows
    if (this._isIgnoredKey(e)) return;

    const key = e.key;
    // Avoid recording repeat events (key held down)
    if (this._pressMap[key] !== undefined) return;

    this._pressMap[key] = performance.now();
  }

  _handleKeyUp(e) {
    if (!this._active) return;
    if (this._isIgnoredKey(e)) return;

    const key = e.key;
    const pressTime = this._pressMap[key];
    if (pressTime === undefined) return;

    delete this._pressMap[key];

    const releaseTime = performance.now();
    const dwell = releaseTime - pressTime;

    // Filter: ignore very long holds (>2000ms = not typing)
    if (dwell > 2000 || dwell < 5) return;

    // PRIVACY: only timing data is recorded/transmitted — never key content.
    // This complies with the project's UK GDPR / ethics commitments:
    // keystroke *content* never leaves the browser.
    this._events.push({
      press_time:   Math.round(pressTime * 100) / 100,
      release_time: Math.round(releaseTime * 100) / 100,
    });

    this._totalKeys++;

    // Emit window when threshold reached
    if (this.onWindow && this._events.length >= this.windowSize) {
      const window = this._events.splice(0, this.windowSize);
      this.onWindow(window);
    }
  }

  _isIgnoredKey(e) {
    const ignored = [
      "Shift", "Control", "Alt", "Meta", "CapsLock", "Tab",
      "Escape", "F1", "F2", "F3", "F4", "F5", "F6", "F7",
      "F8", "F9", "F10", "F11", "F12", "Insert", "Delete",
      "Home", "End", "PageUp", "PageDown", "ArrowUp",
      "ArrowDown", "ArrowLeft", "ArrowRight", "NumLock",
      "ScrollLock", "Pause", "PrintScreen", "ContextMenu"
    ];
    return ignored.includes(e.key);
  }
}
