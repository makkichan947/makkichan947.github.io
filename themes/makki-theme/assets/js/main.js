/* ==========================================================================
   Makki Theme — fluid interactions.
   Springs, not transitions. Interruptible. Velocity handoff. 1:1 drag.
   Rules: respond on press, start from the presentation value, project
   momentum, rubber-band boundaries, respect reduced motion.
   ========================================================================== */
(() => {
  'use strict';

  const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ------------------------------ spring engine ------------------------------ */
  const kFromResponse = (response) => (Math.PI * 2 / Math.max(response, 0.02)) ** 2;
  const drivers = new WeakMap();
  const propsFor = (d, name) => {
    let p = d.props.get(name);
    if (!p) { p = { pos: 0, vel: 0, active: false }; d.props.set(name, p); }
    return p;
  };

  function writeStyle(d) {
    const s = d.el.style;
    const x = d.props.get('x'), y = d.props.get('y'), sc = d.props.get('scale');
    const tr = [];
    if (x && (x.pos || x.active)) tr.push(`translateX(${x.pos}px)`);
    if (y && (y.pos || y.active)) tr.push(`translateY(${y.pos}px)`);
    if (sc && (sc.pos !== 1 || sc.active)) tr.push(`scale(${sc.pos})`);
    s.transform = tr.join(' ') || '';
    const o = d.props.get('opacity');
    if (o) s.opacity = o.pos;
    const b = d.props.get('blur');
    if (b) s.backdropFilter = `blur(${b.pos}px) saturate(170%)`;
    const w = d.props.get('width');
    if (w) s.width = `${w.pos}px`;
  }

  function driver(el) {
    let d = drivers.get(el);
    if (d) return d;
    d = {
      el, props: new Map(), running: false, last: 0,
      frame: null,
    };
    d.frame = () => {
      const dt = Math.min((performance.now() - d.last) / 1000, 1 / 30);
      d.last = performance.now();
      const done = [];
      for (const [name, p] of d.props) {
        if (!p.active) continue;
        /* Implicit damping keeps the integrator stable for stiff springs
           (response 0.12 needs dt < ~9.5ms; explicit Euler diverges at 60Hz
           and long-pressing a button would blow the scale up forever). */
        p.vel = (p.vel - p.k * dt * (p.pos - p.to)) / (1 + p.c * dt);
        p.pos += p.vel * dt;
        if (Math.abs(p.pos - p.to) < 0.02 && Math.abs(p.vel) < 0.02) {
          p.pos = p.to; p.vel = 0; p.active = false;
          if (p.onComplete) done.push(p.onComplete);
        }
      }
      writeStyle(d);
      for (const fn of done) fn();
      if (d.props.size && [...d.props.values()].some((p) => p.active)) {
        d.raf = requestAnimationFrame(d.frame);
      } else {
        d.running = false;
      }
    };
    drivers.set(el, d);
    return d;
  }

  /* Spring one property. Re-targeting an active spring keeps position and
     velocity — interrupts from the presentation value without a jump. */
  function setSpring(el, name, to, opts = {}) {
    if (REDUCED) {
      const d = driver(el);
      const p = propsFor(d, name);
      p.pos = to; p.vel = 0; p.active = false;
      writeStyle(d);
      return;
    }
    const d = driver(el);
    const p = propsFor(d, name);
    const response = opts.response != null ? opts.response : 0.4;
    const damping = opts.damping != null ? opts.damping : 1;
    p.to = to;
    p.k = kFromResponse(response);
    p.c = 2 * damping * Math.sqrt(p.k);
    if (opts.velocity != null) p.vel = opts.velocity;
    if (opts.onComplete) p.onComplete = opts.onComplete;
    p.active = true;
    if (!d.running) {
      d.running = true;
      d.last = performance.now();
      d.raf = requestAnimationFrame(d.frame);
    }
  }

  /* Freeze a spring at its current on-screen value (used when a drag starts). */
  function stopSpring(el, name) {
    const d = driver(el);
    const p = d.props.get(name);
    if (p) { p.active = false; p.vel = 0; }
  }

  /* 1:1 tracking — write directly, no spring (rule: touch and content move together). */
  function setDirect(el, name, v) {
    const d = driver(el);
    const p = propsFor(d, name);
    p.pos = v; p.vel = 0; p.active = false;
    writeStyle(d);
  }

  /* Boundary resistance: real things slow before they stop. */
  const rubberband = (overshoot, dim, k = 0.55) =>
    (overshoot * dim * k) / (dim + k * Math.abs(overshoot));

  const q = (sel, ctx = document) => ctx.querySelector(sel);
  const qa = (sel, ctx = document) => Array.from(ctx.querySelectorAll(sel));

  const IS_REDUCED = () => REDUCED;

  /* ------------------------------ reveal on scroll --------------------------- */
  function initReveal() {
    const els = qa('[data-reveal]');
    if (!els.length || REDUCED) return;
    const io = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        io.unobserve(entry.target);
        /* Capture the CSS offset as the spring's starting value, then kill
           the CSS rule so the resting transform is none — otherwise the card
           jumps and presses slide links out from under the pointer. */
        setDirect(entry.target, 'y', 24);
        entry.target.classList.add('is-revealed');
        const delay = parseInt(entry.target.dataset.revealDelay || '0', 10);
        setTimeout(() => {
          setSpring(entry.target, 'y', 0, { response: 0.5 });
          setSpring(entry.target, 'opacity', 1, { response: 0.45 });
        }, delay);
      }
    }, { threshold: 0.08, rootMargin: '0px 0px -32px 0px' });
    els.forEach((el) => io.observe(el));
  }

  /* --------------------- press / hover physics (delegated) ------------------- */
  function initPress() {
    const hoverState = (el) =>
      !REDUCED && el.classList.contains('card')
        ? { scale: 1.01, y: -3 }
        : null;

    document.addEventListener('pointerover', (e) => {
      const el = e.target.closest('[data-press]');
      if (!el) return;
      el._hovering = true;
      const hs = hoverState(el);
      if (hs) {
        setSpring(el, 'scale', hs.scale, { response: 0.5 });
        setSpring(el, 'y', hs.y, { response: 0.5 });
      }
    });

    document.addEventListener('pointerout', (e) => {
      const el = e.target.closest('[data-press]');
      if (!el || el._pressed || !el._hovering) return;
      if (e.relatedTarget && el.contains(e.relatedTarget)) return;
      el._hovering = false;
      setSpring(el, 'scale', 1, { response: 0.35 });
      setSpring(el, 'y', 0, { response: 0.35 });
    });

    /* Feedback lives on pointer-down — instant, not on release. */
    document.addEventListener('pointerdown', (e) => {
      const el = e.target.closest('[data-press]');
      if (!el) return;
      el._pressed = true;
      setSpring(el, 'scale', 0.97, { response: 0.12 });
    });

    ['pointerup', 'pointercancel'].forEach((ev) => {
      document.addEventListener(ev, (e) => {
        const el = e.target.closest('[data-press]');
        if (!el || !el._pressed) return;
        el._pressed = false;
        const hs = hoverState(el);
        setSpring(el, 'scale', hs ? hs.scale : 1, { response: 0.3 });
        setSpring(el, 'y', hs ? hs.y : 0, { response: 0.3 });
      });
    });
  }

  /* ------------------------------- navbar ------------------------------------ */
  function initNav() {
    const navbar = q('.navbar');
    const tick = q('.nav-tick');
    const wrap = q('.nav-menu-wrap');

    let scrollTicking = false;
    const onScroll = () => {
      if (scrollTicking) return;
      scrollTicking = true;
      requestAnimationFrame(() => {
        scrollTicking = false;
        if (navbar) navbar.classList.toggle('scrolled', window.scrollY > 8);
      });
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    /* Neon tick springs between links. */
    if (wrap && tick) {
      const place = () => {
        const link = q('.nav-menu a.is-active', wrap);
        if (!link) {
          setSpring(tick, 'opacity', 0, { response: 0.3 });
          return;
        }
        const mr = wrap.getBoundingClientRect();
        const r = link.getBoundingClientRect();
        const w = r.width * 0.6;
        setSpring(tick, 'x', r.left - mr.left + r.width * 0.2, { response: 0.45 });
        setSpring(tick, 'width', w, { response: 0.45 });
        setSpring(tick, 'opacity', 1, { response: 0.3 });
      };
      place();
      window.addEventListener('resize', place);
      window.addEventListener('load', place);
      setTimeout(place, 500); /* re-measure after fonts */
    }
  }

  /* ------------------------------- back to top -------------------------------- */
  function initBackToTop() {
    const btn = q('.back-to-top');
    if (!btn) return;
    const k = kFromResponse(0.5);
    const c = 2 * Math.sqrt(k);
    let visible = false;
    let vScroll = 0, lastY = window.scrollY, lastT = performance.now();
    let raf = 0;

    const show = () => {
      if (visible) return;
      visible = true;
      btn.style.pointerEvents = 'auto';
      setSpring(btn, 'opacity', 1, { response: 0.35 });
      setSpring(btn, 'scale', 1, { response: 0.4, damping: 0.9 });
    };
    const hide = () => {
      if (!visible) return;
      visible = false;
      setSpring(btn, 'opacity', 0, { response: 0.3 });
      setSpring(btn, 'scale', 0.8, { response: 0.3, onComplete: () => {
        if (!visible) btn.style.pointerEvents = 'none';
      } });
    };

    window.addEventListener('scroll', () => {
      const now = performance.now();
      const dy = window.scrollY - lastY;
      const dt = now - lastT;
      if (dt > 0) vScroll = dy / (dt / 1000);
      lastY = window.scrollY;
      lastT = now;
      if (window.scrollY > window.innerHeight * 0.8) show();
      else hide();
    }, { passive: true });

    /* Hand off the finger's scroll velocity into the spring. */
    btn.addEventListener('click', () => {
      if (REDUCED) { window.scrollTo({ top: 0, behavior: 'smooth' }); return; }
      cancelAnimationFrame(raf);
      let y = window.scrollY, v = Math.max(0, Math.min(vScroll, 2400));
      const step = () => {
        const dt = 1 / 60;
        const a = -k * (y - 0) - c * v;
        v += a * dt;
        y += v * dt;
        if (Math.abs(y) < 1 && Math.abs(v) < 1) { window.scrollTo(0, 0); return; }
        window.scrollTo(0, Math.max(0, y));
        raf = requestAnimationFrame(step);
      };
      step();
    });
  }

  /* ------------------------------ search sheet ------------------------------- */
  const sheetRef = { open: false };

  function initSheet() {
    const sheet = q('.sheet');
    const panel = q('.sheet-panel');
    const scrim = q('.sheet-scrim');
    const handle = q('.sheet-handle');
    const input = q('#search-input');
    const toggle = q('#search-toggle');
    if (!sheet || !panel || !scrim) return;

    const vh = () => window.innerHeight;
    let drag = null;

    /* Presentation values — the sheet hangs below the viewport by default. */
    setDirect(panel, 'y', vh() * 1.05);
    setDirect(panel, 'opacity', 0);
    setDirect(panel, 'scale', 0.94);
    setDirect(panel, 'blur', 0);
    setDirect(scrim, 'opacity', 0);

    const commitClosed = () => {
      if (sheetRef.open) return;
      sheet.classList.remove('is-open');
      scrim.classList.remove('is-open');
      sheet.setAttribute('aria-hidden', 'true');
      if (toggle) toggle.setAttribute('aria-expanded', 'false');
      document.body.classList.remove('lock');
    };

    const openSheet = () => {
      if (sheetRef.open) return;
      sheetRef.open = true;
      document.body.classList.add('lock');
      sheet.classList.add('is-open');
      scrim.classList.add('is-open');
      sheet.setAttribute('aria-hidden', 'false');
      if (toggle) toggle.setAttribute('aria-expanded', 'true');
      setSpring(panel, 'y', 0, { response: 0.38, damping: 1 });
      setSpring(panel, 'opacity', 1, { response: 0.3 });
      setSpring(panel, 'scale', 1, { response: 0.42, damping: 0.9 });
      setSpring(panel, 'blur', 32, { response: 0.35 });
      setSpring(scrim, 'opacity', 1, { response: 0.3 });
      if (input) input.focus({ preventScroll: true });
      document.dispatchEvent(new CustomEvent('makki:sheet-open'));
    };

    /* Same path out as in — springs back down, velocity handed off. */
    const closeSheet = (velocity = 0) => {
      if (!sheetRef.open) return;
      sheetRef.open = false;
      setSpring(panel, 'y', vh() * 1.05, { response: 0.38, damping: 1, velocity });
      setSpring(panel, 'opacity', 0, { response: 0.26 });
      setSpring(panel, 'scale', 0.94, { response: 0.28 });
      setSpring(panel, 'blur', 0, { response: 0.26 });
      setSpring(scrim, 'opacity', 0, { response: 0.28, onComplete: commitClosed });
      if (toggle) toggle.focus({ preventScroll: true });
      document.dispatchEvent(new CustomEvent('makki:sheet-close'));
    };

    const toggleSheet = () => (sheetRef.open ? closeSheet() : openSheet());

    if (toggle) toggle.addEventListener('click', () => toggleSheet());

    /* Esc and the backdrop close; Ctrl+K or "/" open. */
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && sheetRef.open) { closeSheet(); return; }
      const mod = e.ctrlKey || e.metaKey;
      const target = e.target;
      const typing = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA');
      if ((mod && e.key.toLowerCase() === 'k') || (e.key === '/' && !typing)) {
        e.preventDefault();
        openSheet();
      }
    });

    if (scrim) scrim.addEventListener('click', (e) => {
      if (e.target === scrim) closeSheet();
    });

    /* --- drag: 1:1 tracking, velocity history, momentum projection,
          rubber-banding, fully interruptible mid-flight. --- */
    if (handle && !REDUCED) {
      const onDown = (e) => {
        if (e.button > 0 || e.target.closest('input, a, button, .search-results')) return;
        handle.setPointerCapture(e.pointerId);
        stopSpring(panel, 'y');
        drag = {
          pointerId: e.pointerId,
          startY: e.clientY,
          y: 0,
          samples: [],
        };
        handle.classList.add('is-dragging');
      };

      const onMove = (e) => {
        if (!drag || e.pointerId !== drag.pointerId) return;
        const raw = e.clientY - drag.startY;
        /* Rubber-band only past the open position (pulling up). */
        const y = raw < 0 ? -rubberband(-raw, vh()) : raw;
        drag.y = y;
        setDirect(panel, 'y', y);
        const now = performance.now();
        drag.samples.push({ t: now, y });
        if (drag.samples.length > 8) drag.samples.shift();
      };

      const onUp = (e) => {
        if (!drag || e.pointerId !== drag.pointerId) return;
        const samples = drag.samples;
        let v = 0;
        if (samples.length >= 2) {
          const first = samples[0];
          const last = samples[samples.length - 1];
          const dt = (last.t - first.t) / 1000;
          if (dt > 0.001) v = (last.y - first.y) / dt;
        }
        const y = drag.y;
        handle.classList.remove('is-dragging');
        drag = null;
        /* Where is the gesture going? Project, then choose the target.
           Commit or reverse by velocity sign — never by release position alone. */
        const projected = y + v * 0.35;
        if (projected > vh() * 0.3 || v > 1500) {
          sheetRef.open = false;
          setSpring(panel, 'y', vh() * 1.05, { response: 0.38, damping: 1, velocity: Math.min(v, 2200) });
          setSpring(panel, 'opacity', 0, { response: 0.26 });
          setSpring(panel, 'scale', 0.94, { response: 0.28 });
          setSpring(panel, 'blur', 0, { response: 0.26 });
          setSpring(scrim, 'opacity', 0, { response: 0.28, onComplete: commitClosed });
          if (toggle) toggle.focus({ preventScroll: true });
          document.dispatchEvent(new CustomEvent('makki:sheet-close'));
        } else {
          /* Snap back home — the spring continues at the finger's velocity. */
          setSpring(panel, 'y', 0, { response: 0.4, damping: 1, velocity: v });
          setSpring(panel, 'scale', 1, { response: 0.35 });
          setSpring(panel, 'blur', 32, { response: 0.3 });
          setSpring(scrim, 'opacity', 1, { response: 0.25 });
        }
      };

      handle.addEventListener('pointerdown', onDown);
      handle.addEventListener('pointermove', onMove);
      handle.addEventListener('pointerup', onUp);
      handle.addEventListener('pointercancel', onUp);
    }

    window.addEventListener('resize', () => {
      if (sheetRef.open) setSpring(panel, 'y', 0, { response: 0.35 });
    });

    /* Focus trap inside the sheet. */
    sheet.addEventListener('keydown', (e) => {
      if (e.key !== 'Tab') return;
      const f = qa('input, a, button', sheet).filter((el) => el.offsetParent !== null);
      if (!f.length) return;
      const first = f[0];
      const last = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    });

    window.Makki = Object.assign(window.Makki || {}, { openSearch: openSheet, closeSearch: closeSheet, toggleSearch: toggleSheet, sheetOpen: () => sheetRef.open });
  }

  /* --------------------------------- TOC ------------------------------------- */
  function initToc() {
    const toc = q('.toc');
    if (!toc) return;
    const links = qa('.toc a');
    const ids = links.map((l) => l.getAttribute('href'));
    const heads = ids.map((id) => id && q(id)).filter(Boolean);
    if (!heads.length) return;
    const navH = q('.navbar') ? q('.navbar').offsetHeight : 0;
    let ticking = false;

    const mark = () => {
      ticking = false;
      const y = window.scrollY + navH + 16;
      let current = 0;
      heads.forEach((h, i) => { if (h.getBoundingClientRect().top + window.scrollY <= y) current = i; });
      links.forEach((l, i) => l.classList.toggle('is-current', i === current));
    };

    window.addEventListener('scroll', () => {
      if (!ticking) { ticking = true; requestAnimationFrame(mark); }
    }, { passive: true });
    mark();
  }

  /* ---------------------------------- copy ----------------------------------- */
  function initCopy() {
    qa('.article-content pre code').forEach((block) => {
      const pre = block.parentElement;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'copy-button';
      btn.textContent = '复制';
      btn.setAttribute('data-press', '');
      btn.addEventListener('click', () => {
        navigator.clipboard.writeText(block.textContent).then(() => {
          btn.classList.add('is-copied');
          btn.textContent = '已复制';
          clearTimeout(btn._t);
          btn._t = setTimeout(() => {
            btn.classList.remove('is-copied');
            btn.textContent = '复制';
          }, 2000);
        });
      });
      pre.appendChild(btn);
    });
  }

  /* -------------------------------- math ------------------------------------- */
  function initMath() {
    if (typeof renderMathInElement === 'function') {
      renderMathInElement(q('.article-content') || document.body, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false },
        ],
      });
    }
  }

  /* --------------------------------- boot ------------------------------------ */
  document.addEventListener('DOMContentLoaded', () => {
    initReveal();
    initPress();
    initNav();
    initBackToTop();
    initSheet();
    initToc();
    initCopy();
    initMath();
  });
})();
