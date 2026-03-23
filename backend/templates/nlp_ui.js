/**
 * nlp_ui.js — NLP Resume Enhancement Frontend Module
 * =====================================================
 * Handles rendering of:
 *  - Skill Gap Dashboard (strong / partial / missing)
 *  - Section Scores (summary, experience, skills, education)
 *  - ATS Compliance Panel
 *  - Change Stats (bullets improved, metrics added)
 *  - Before/After diff tab
 *
 * Include this script BEFORE the main app script in index.html.
 * Exposes: window.NLP = { renderDashboard, generateRoleResume }
 */

(function (global) {
  'use strict';

  /* ─── helpers ─────────────────────────────────────────── */
  const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

  function scoreColor(score) {
    if (score >= 80) return '#10b981';   // green
    if (score >= 60) return '#f59e0b';   // amber
    if (score >= 40) return '#ef4444';   // red
    return '#7f1d1d';
  }
  function gradeBadge(grade) {
    const colors = { A:'#10b981', B:'#3b82f6', C:'#f59e0b', D:'#ef4444', F:'#7f1d1d' };
    const c = colors[grade] || '#64748b';
    return `<span style="background:${c};color:#fff;border-radius:4px;padding:1px 7px;font-size:.72em;font-weight:700;">${esc(grade)}</span>`;
  }

  /* ─── skill gap section ────────────────────────────────── */
  function renderSkillGap(gap) {
    if (!gap) return '';
    const scoreBar = `
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
        <div style="flex:1;height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
          <div style="height:100%;width:${gap.score}%;background:${scoreColor(gap.score)};border-radius:4px;transition:width .6s;"></div>
        </div>
        <span style="font-size:.8em;font-weight:700;color:${scoreColor(gap.score)};">${gap.score}/100</span>
      </div>`;

    const renderPills = (items, color, icon) =>
      items.length ? items.map(s =>
        `<span style="display:inline-block;margin:2px 3px;padding:2px 9px;border-radius:12px;font-size:.73em;font-weight:600;background:${color}20;color:${color};border:1px solid ${color}40;">${icon} ${esc(s)}</span>`
      ).join('') : `<span style="color:#94a3b8;font-size:.75em;">None detected</span>`;

    return `
      <div class="nlp-gap-block">
        <div style="font-size:.78em;font-weight:700;color:#64748b;letter-spacing:.8px;text-transform:uppercase;margin-bottom:6px;">
          🎯 Skill Fit — ${esc(gap.role_title)} &nbsp;·&nbsp; ${gap.coverage}% required coverage
        </div>
        ${scoreBar}
        <div style="margin-bottom:5px;font-size:.74em;font-weight:700;color:#10b981;">✅ Strong (${gap.strong.length})</div>
        <div style="margin-bottom:8px;">${renderPills(gap.strong, '#10b981', '')}</div>
        ${gap.partial.length ? `<div style="margin-bottom:5px;font-size:.74em;font-weight:700;color:#f59e0b;">⚠ Partial (${gap.partial.length})</div><div style="margin-bottom:8px;">${renderPills(gap.partial, '#f59e0b', '')}</div>` : ''}
        <div style="margin-bottom:5px;font-size:.74em;font-weight:700;color:#ef4444;">❌ Missing Required (${gap.missing_required.length})</div>
        <div>${renderPills(gap.missing_required, '#ef4444', '')}</div>
      </div>`;
  }

  /* ─── section scores ───────────────────────────────────── */
  function renderSectionScores(scores) {
    if (!scores) return '';
    const icons = { summary:'📝', experience:'💼', skills:'🛠', education:'🎓' };
    const rows = Object.entries(scores).map(([key, val]) => {
      const { score, grade, strengths=[], weaknesses=[], fixes=[] } = val;
      const id = `nlp-sec-${key}-${Math.random().toString(36).slice(2,7)}`;
      return `
        <div style="border:1px solid #e2e8f0;border-radius:10px;padding:10px 13px;margin-bottom:8px;">
          <div style="display:flex;align-items:center;gap:8px;cursor:pointer;"
               onclick="document.getElementById('${id}').style.display=document.getElementById('${id}').style.display==='none'?'block':'none'">
            <span style="font-size:.9em;">${icons[key]||'📄'}</span>
            <span style="font-size:.82em;font-weight:700;flex:1;text-transform:capitalize;">${esc(key)}</span>
            ${gradeBadge(grade)}
            <div style="flex:1;height:6px;background:#e2e8f0;border-radius:3px;max-width:80px;overflow:hidden;">
              <div style="height:100%;width:${score}%;background:${scoreColor(score)};border-radius:3px;"></div>
            </div>
            <span style="font-size:.76em;color:${scoreColor(score)};font-weight:700;">${score}</span>
            <span style="font-size:.7em;color:#94a3b8;">▾</span>
          </div>
          <div id="${id}" style="display:none;margin-top:8px;font-size:.75em;">
            ${strengths.length ? `<div style="color:#065f46;margin-bottom:4px;">${strengths.map(s=>`✓ ${esc(s)}`).join('<br>')}</div>` : ''}
            ${weaknesses.length ? `<div style="color:#991b1b;margin-bottom:4px;">${weaknesses.map(s=>`✗ ${esc(s)}`).join('<br>')}</div>` : ''}
            ${fixes.length ? `<div style="color:#92400e;">${fixes.map(s=>`→ ${esc(s)}`).join('<br>')}</div>` : ''}
          </div>
        </div>`;
    });
    return `<div class="nlp-scores-block"><div style="font-size:.78em;font-weight:700;color:#64748b;letter-spacing:.8px;text-transform:uppercase;margin-bottom:8px;">📊 Section Quality Scores</div>${rows.join('')}</div>`;
  }

  /* ─── ATS panel ────────────────────────────────────────── */
  function renderAts(ats) {
    if (!ats) return '';
    const bar = `
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <div style="flex:1;height:10px;background:#e2e8f0;border-radius:5px;overflow:hidden;">
          <div style="height:100%;width:${ats.score}%;background:${scoreColor(ats.score)};border-radius:5px;transition:width .6s;"></div>
        </div>
        <span style="font-size:.82em;font-weight:700;color:${scoreColor(ats.score)};">${ats.score}/100 ${esc(ats.grade)}</span>
      </div>`;

    const issueHtml = ats.issues.length
      ? `<div style="color:#991b1b;font-size:.74em;margin-top:6px;">${ats.issues.map(i=>`⚠ ${esc(i)}`).join('<br>')}</div>` : '';
    const passHtml  = ats.passed.length
      ? `<div style="color:#065f46;font-size:.74em;margin-top:4px;">${ats.passed.slice(0,4).map(p=>`✓ ${esc(p)}`).join('<br>')}</div>` : '';

    return `
      <div class="nlp-ats-block">
        <div style="font-size:.78em;font-weight:700;color:#64748b;letter-spacing:.8px;text-transform:uppercase;margin-bottom:6px;">
          🤖 ATS Score (Enhanced)
        </div>
        ${bar}
        ${issueHtml}
        ${passHtml}
      </div>`;
  }

  /* ─── stats strip ──────────────────────────────────────── */
  function renderStats(stats, changeLog) {
    if (!stats) return '';
    const tiles = [
      { icon:'⚡', label:'Bullets Improved', val: stats.bullets_improved },
      { icon:'📈', label:'Metrics Added',    val: `${stats.metrics_before}→${stats.metrics_after}` },
      { icon:'🎯', label:'Skills Matched',   val: stats.skills_matched },
      { icon:'❌', label:'Gaps Found',        val: stats.skills_missing },
    ];
    return `
      <div class="nlp-stats-block">
        <div style="font-size:.78em;font-weight:700;color:#64748b;letter-spacing:.8px;text-transform:uppercase;margin-bottom:8px;">⚙ NLP Enhancement Stats</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
          ${tiles.map(t => `
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:7px 10px;text-align:center;">
              <div style="font-size:1.2em;">${t.icon}</div>
              <div style="font-size:1em;font-weight:800;color:#1a1a2e;">${esc(String(t.val))}</div>
              <div style="font-size:.7em;color:#64748b;">${t.label}</div>
            </div>`).join('')}
        </div>
        ${changeLog && changeLog.length ? `<div style="font-size:.72em;color:#64748b;margin-top:6px;">Changes: ${[...new Set(changeLog)].map(c=>`<span style="background:#f1f5f9;border-radius:4px;padding:1px 5px;margin:0 2px;">${esc(c)}</span>`).join('')}</div>` : ''}
      </div>`;
  }

  /* ─── MASTER RENDER ────────────────────────────────────── */
  /**
   * renderNlpDashboard(cardIdx, nlpData)
   * Injects the full NLP dashboard below the summary preview in card `cardIdx`.
   */
  function renderNlpDashboard(cardIdx, nlpData) {
    if (!nlpData) return;

    // Find or create the dashboard container inside the card
    let container = document.getElementById(`nlp-dash-${cardIdx}`);
    if (!container) {
      const card = document.getElementById(`rpc-${cardIdx}`);
      if (!card) return;
      container = document.createElement('div');
      container.id = `nlp-dash-${cardIdx}`;
      container.style.cssText = 'margin-top:12px;display:flex;flex-direction:column;gap:10px;';
      // Insert after the rpc-actions div
      const actions = card.querySelector('.rpc-actions');
      if (actions && actions.parentNode) {
        actions.parentNode.insertBefore(container, actions.nextSibling);
      } else {
        card.querySelector('.rpc-body').appendChild(container);
      }
    }

    const { skill_gap, section_scores, ats_result, stats, change_log } = nlpData;

    container.innerHTML = `
      <div style="border-top:1px solid #e2e8f0;padding-top:12px;">
        ${renderStats(stats, change_log)}
      </div>
      <div>${renderSkillGap(skill_gap)}</div>
      <div>${renderSectionScores(section_scores)}</div>
      <div>${renderAts(ats_result)}</div>`;
  }

  /* ─── export ───────────────────────────────────────────── */
  global.NLP = { renderNlpDashboard };

}(window));