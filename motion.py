"""The opening sequence: tells the page when its first content is ready.

app_v2.py shows a branded veil over the page while the first run builds it.
This tiny component sits at the end of the page, so it mounts once
everything above it exists. It then adds two classes to the page:
  hr-ready   - lift the veil and let the blocks rise in, one after another
  hr-settled - from then on, new blocks (another view, a new chart) fade in quickly
If it never runs, the CSS lifts the veil by itself after a few seconds.
"""
import streamlit as st

JS = r"""
export default function () {
  if (window.__headroomMotion) return;
  window.__headroomMotion = true;
  const root = document.documentElement;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    root.classList.add("hr-ready");
    setTimeout(() => root.classList.add("hr-settled"), 1900);
  }));
}
"""

# The version in the name makes browsers load this code fresh instead of a cached older copy.
_motion = st.components.v2.component("headroom_motion_v1", js=JS)


def start_motion():
    """Call once, at the very end of the page."""
    _motion(key="hr_motion", data={"v": 1})
