"use strict";
(() => {
  const data = window.EVO_RUN_INVENTORY;
  if (!data) {
    document.getElementById("count").textContent = "The local inventory asset is missing. Use the linked static analyses or regenerate the inventory.";
    return;
  }
  const fmt = value => Number(value).toLocaleString();
  document.getElementById("stat-groups").textContent = fmt(data.totals.families);
  document.getElementById("stat-logs").textContent = fmt(data.totals.jsonl_files);
  document.getElementById("stat-rows").textContent = fmt(data.totals.jsonl_rows);
  document.getElementById("snapshot").textContent = `Snapshot ${data.generated_utc.slice(0,10)} · ${fmt(data.totals.files)} local files inventoried · ${fmt(data.totals.malformed_rows)} malformed JSONL rows · raw conversations omitted`;
  const task = document.getElementById("task");
  for (const name of [...new Set(data.logs.map(row => row.task))].sort()) {
    const option = document.createElement("option"); option.value = name;
    option.textContent = name === "other" ? "Other / research-specific schema" : name;
    task.append(option);
  }
  const query = document.getElementById("query"), status = document.getElementById("status"), sort = document.getElementById("sort");
  const cell = (row, content) => { const td = document.createElement("td"); if (typeof content === "string") td.textContent = content; else td.append(content); row.append(td); };
  function render() {
    const text = query.value.trim().toLowerCase();
    const rows = data.logs.filter(row => (!task.value || row.task === task.value)
      && (!status.value || (status.value === "ended") === row.has_run_end)
      && (!text || [row.source, ...Object.keys(row.models), ...Object.keys(row.events)].join(" ").toLowerCase().includes(text)));
    rows.sort((a,b) => sort.value === "rows" ? b.rows-a.rows : sort.value === "recent"
      ? (b.last_event_utc || "").localeCompare(a.last_event_utc || "") : a.source.localeCompare(b.source));
    const body = document.getElementById("runs"); body.replaceChildren();
    for (const row of rows) {
      const tr = document.createElement("tr"), details = document.createElement("details"), summary = document.createElement("summary");
      summary.textContent = row.source.replace(/^(?:\.local\/)?runs\//, ""); details.append(summary);
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify({events:row.events, models:row.models, config:row.config || null,
        recorded_summary:row.summary || null, malformed_rows:row.malformed_rows, sha256:row.sha256}, null, 2);
      details.append(pre);
      if (/^(?:\.local\/)?runs\//.test(row.source) && !row.source.split("/").includes("..")) {
        const link = document.createElement("a"); link.href = "../" + (row.source.startsWith(".local/") ? "" : ".local/") + row.source.split("/").map(encodeURIComponent).join("/");
        link.textContent = "Open raw source (local checkout) ↗"; details.append(link);
      }
      cell(tr,details); cell(tr,row.task); cell(tr,fmt(row.rows));
      const badge = document.createElement("span"); badge.className = "pill" + (row.has_run_end ? "" : " muted");
      badge.textContent = row.has_run_end ? "End recorded" : "No run_end";
      cell(tr,badge); cell(tr,row.last_event_utc ? row.last_event_utc.slice(0,10) : "Not recorded"); body.append(tr);
    }
    document.getElementById("count").textContent = `${fmt(rows.length)} of ${fmt(data.logs.length)} JSONL files · expand a source to inspect configuration, events and summary`;
    document.getElementById("empty").hidden = rows.length > 0;
  }
  for (const field of [query,task,status,sort]) field.addEventListener("input",render);
  render();
})();
