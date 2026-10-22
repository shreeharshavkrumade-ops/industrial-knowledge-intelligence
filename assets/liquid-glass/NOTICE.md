# Bundled third-party assets
- `container.js`, `glass.css` - from https://github.com/dashersw/liquid-glass-js (MIT, (c) 2025 Armagan Amcalar). Unmodified.
- `html2canvas.min.js` - html2canvas 1.4.1 (MIT, (c) Niklas von Hertzen), required by Liquid Glass JS for page sampling. Unmodified.
They are inlined into a Streamlit component iframe by `src/glass_component.py`; no CDN or network access is needed.
