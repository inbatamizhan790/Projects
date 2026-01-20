// result.js - set dynamic badge color based on AOI
document.addEventListener('DOMContentLoaded', function() {
  const badge = document.getElementById('aoi-badge');
  if (!badge) return;
  const aoi = parseFloat(badge.dataset.aoi);
  // Map AOI 0..90 to 0..1
  const t = Math.min(Math.max(aoi / 90.0, 0.0), 1.0);

  // Color gradient: green -> yellow -> red
  function lerpColor(c1, c2, f){
    const r = Math.round(c1[0] + (c2[0]-c1[0])*f);
    const g = Math.round(c1[1] + (c2[1]-c1[1])*f);
    const b = Math.round(c1[2] + (c2[2]-c1[2])*f);
    return `rgb(${r}, ${g}, ${b})`;
  }
  const green = [34,197,94];   // #22c55e
  const yellow = [250,204,21]; // #facd15
  const red = [239,68,68];     // #ef4444

  let color;
  if (t < 0.5) {
    color = lerpColor(green, yellow, t*2);
  } else {
    color = lerpColor(yellow, red, (t-0.5)*2);
  }

  // Apply as glow and small gradient
  badge.style.boxShadow = `0 8px 30px ${color}40, inset 0 -2px 8px ${color}20`;
  // create a small top border highlight
  badge.style.borderTop = `4px solid ${color}`;
  // color the numeric value slightly
  const val = badge.querySelector('.aoi-value');
  if (val) {
    val.style.color = color;
    val.style.transition = 'color 500ms ease';
  }

  // small pulsate animation based on AOI
  const scale = 1 + 0.03 * (t*2); // higher AOI -> slightly bigger pulse
  badge.animate([
    { transform: 'scale(1)' },
    { transform: `scale(${scale})` },
    { transform: 'scale(1)' }
  ], { duration: 1400, iterations: 2, easing: 'ease-in-out' });
});
