// Renderer: talks to the local Python API (loaded once by the main process).
const API = "http://127.0.0.1:8765";

const $ = (id) => document.getElementById(id);

async function api(path) {
  const res = await fetch(API + path);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "request failed");
  return data;
}

// Wait for the engine, then enable the UI and load teams.
async function waitForReady() {
  for (let i = 0; i < 60; i++) {
    try {
      await api("/health");
      return true;
    } catch {
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
  return false;
}

function fillSelect(sel, teams, preferred) {
  sel.innerHTML = "";
  for (const t of teams) {
    const o = document.createElement("option");
    o.value = t;
    o.textContent = t;
    sel.appendChild(o);
  }
  if (preferred && teams.includes(preferred)) sel.value = preferred;
}

async function init() {
  // Tab switching.
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      $(tab.dataset.tab).classList.add("active");
    });
  });

  const ready = await waitForReady();
  if (!ready) {
    $("status").textContent = "engine failed to start";
    return;
  }
  $("status").textContent = "ready";

  const teams = await api("/teams");
  fillSelect($("home"), teams.all, "Argentina");
  fillSelect($("away"), teams.all, "Brazil");
  $("predict").disabled = false;
  $("simulate").disabled = false;

  $("predict").addEventListener("click", onPredict);
  $("simulate").addEventListener("click", onSimulate);
}

async function onPredict() {
  const home = $("home").value;
  const away = $("away").value;
  if (home === away) {
    $("status").textContent = "pick two different teams";
    return;
  }
  $("status").textContent = "predicting…";
  const neutral = $("neutral").checked ? 1 : 0;
  const d = await api(
    `/predict?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}&neutral=${neutral}`
  );
  $("status").textContent = "ready";

  $("lh").textContent = `${home} win`;
  $("la").textContent = `${away} win`;
  const set = (bar, pct, v) => {
    $(bar).style.width = `${(v * 100).toFixed(1)}%`;
    $(pct).textContent = `${(v * 100).toFixed(1)}%`;
  };
  set("bh", "ph", d.p_home_win);
  set("bd", "pd", d.p_draw);
  set("ba", "pa", d.p_away_win);
  $("score").textContent = d.score;
  $("conf").textContent = d.confidence;
  $("result").classList.remove("hidden");
}

async function onSimulate() {
  const n = parseInt($("nsims").value, 10) || 2000;
  $("simulate").disabled = true;
  $("simstatus").textContent = `running ${n.toLocaleString()} simulations…`;
  try {
    const rows = await api(`/simulate?n=${n}`);
    const body = $("simtable").querySelector("tbody");
    body.innerHTML = "";
    for (const r of rows.slice(0, 24)) {
      const tr = document.createElement("tr");
      const cells = [
        r.team,
        r.group,
        pct(r.reach_R16),
        pct(r.reach_QF),
        pct(r.reach_SF),
        pct(r.reach_final),
        pct(r.win),
      ];
      cells.forEach((c, i) => {
        const td = document.createElement("td");
        td.textContent = c;
        tr.appendChild(td);
      });
      body.appendChild(tr);
    }
    $("simtable").classList.remove("hidden");
    $("simstatus").textContent = `done (${n.toLocaleString()} sims)`;
  } catch (e) {
    $("simstatus").textContent = `error: ${e.message}`;
  } finally {
    $("simulate").disabled = false;
  }
}

const pct = (v) => `${(v * 100).toFixed(1)}`;

init();
