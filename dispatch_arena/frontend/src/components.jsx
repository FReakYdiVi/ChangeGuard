// Visual components for the dispatch frontend.
//
// The map is plain SVG (no external graph lib) — sufficient for ~10 nodes and
// gives us full control over courier animations. Couriers are rendered as
// foreign objects whose (x, y) interpolates between their previous and target
// node when in transit, driven by `eta_remaining` and `step_progress`.

import React from 'react';
import { STATUS_COLORS, COURIER_STATUS_LABELS } from './fixtures.js';

// Helper: find a node by id
const nodeById = (nodes, id) => nodes.find((n) => n.id === id);

// Distribute couriers so they don't stack on top of each other when several
// share a location (e.g. all 5 couriers at the hub at t=0, or two heading to
// the same store). Returns a Map<courier_id, {x, y}>.
//
// Two cluster types:
//   - "node":  multiple couriers at the same node          -> fan out in a circle
//   - "edge":  multiple couriers traversing the same edge  -> fan out perpendicular
//              to the edge direction (so they look like a row of bikes)
//
// Couriers in transit are rendered slightly past the midpoint of their edge
// (t=0.55) so the eye reads them as "on the way" rather than at either endpoint.
function distributeCouriers(couriers, nodes) {
  const positions = new Map();
  const groups = new Map();

  for (const c of couriers) {
    const here = nodeById(nodes, c.node_id);
    if (!here) continue;

    let entry;
    if (c.target_node_id && c.eta_remaining > 0 && c.status !== 'idle') {
      const there = nodeById(nodes, c.target_node_id);
      if (there) {
        const t = 0.55;
        const centerX = here.x + (there.x - here.x) * t;
        const centerY = here.y + (there.y - here.y) * t;
        const dx = there.x - here.x;
        const dy = there.y - here.y;
        const len = Math.hypot(dx, dy) || 1;
        const normal = { x: -dy / len, y: dx / len };
        entry = {
          key: `edge:${c.node_id}->${c.target_node_id}`,
          type: 'edge',
          centerX, centerY, normal,
        };
      }
    }
    if (!entry) {
      entry = {
        key: `node:${c.node_id}`,
        type: 'node',
        centerX: here.x,
        centerY: here.y,
      };
    }

    if (!groups.has(entry.key)) groups.set(entry.key, { ...entry, members: [] });
    groups.get(entry.key).members.push(c);
  }

  for (const group of groups.values()) {
    const { centerX, centerY, members, type } = group;
    if (members.length === 1) {
      positions.set(members[0].id, { x: centerX, y: centerY });
      continue;
    }
    if (type === 'edge') {
      const spacing = 18;
      members.forEach((c, i) => {
        const offset = (i - (members.length - 1) / 2) * spacing;
        positions.set(c.id, {
          x: centerX + offset * group.normal.x,
          y: centerY + offset * group.normal.y,
        });
      });
    } else {
      // node cluster: fan around the node center
      const radius = 26;
      members.forEach((c, i) => {
        const angle = (2 * Math.PI * i) / members.length - Math.PI / 2;
        positions.set(c.id, {
          x: centerX + radius * Math.cos(angle),
          y: centerY + radius * Math.sin(angle),
        });
      });
    }
  }
  return positions;
}

// ───────────────────────────────────────────────────────────────
// Map
// ───────────────────────────────────────────────────────────────

export function ScenarioMap({ nodes, snapshot, dimmed }) {
  if (!snapshot) return null;
  const { couriers, orders } = snapshot;

  // For every active assignment, draw a thin link from courier to its target.
  const courierLinks = couriers
    .filter((c) => c.target_node_id && c.eta_remaining > 0)
    .map((c) => ({
      from: nodeById(nodes, c.node_id),
      to:   nodeById(nodes, c.target_node_id),
      courier: c,
    }))
    .filter((l) => l.from && l.to);

  return (
    <svg viewBox="0 0 900 520" className={`map-svg${dimmed ? ' dimmed' : ''}`}>
      {/* subtle dotted background */}
      <defs>
        <pattern id="dots" x="0" y="0" width="24" height="24" patternUnits="userSpaceOnUse">
          <circle cx="2" cy="2" r="1" fill="#2a3142" opacity="0.7" />
        </pattern>
      </defs>
      <rect width="900" height="520" fill="url(#dots)" />

      {/* base edges: lighter lines from hub to every store and store to nearest customers */}
      {nodes.filter((n) => n.kind === 'pickup').map((store) => {
        const hub = nodeById(nodes, 'hub');
        return (
          <line key={`hub-${store.id}`}
            x1={hub.x} y1={hub.y} x2={store.x} y2={store.y}
            stroke="#2c3548" strokeWidth="1" strokeDasharray="3 4" />
        );
      })}

      {/* live courier links (overlay) */}
      {courierLinks.map((l, i) => (
        <line key={`link-${i}`}
          x1={l.from.x} y1={l.from.y} x2={l.to.x} y2={l.to.y}
          stroke="#5DA9F0" strokeOpacity="0.6" strokeWidth="2" strokeDasharray="6 4">
          <animate attributeName="stroke-dashoffset" from="0" to="-20" dur="1s" repeatCount="indefinite" />
        </line>
      ))}

      {/* nodes */}
      {nodes.map((n) => <NodeIcon key={n.id} node={n} orders={orders} />)}

      {/* couriers (rendered last so they sit on top) */}
      {(() => {
        const positions = distributeCouriers(couriers, nodes);
        return couriers.map((c) => {
          const pos = positions.get(c.id) || { x: 0, y: 0 };
          return <CourierIcon key={c.id} courier={c} x={pos.x} y={pos.y} orders={orders} />;
        });
      })()}
    </svg>
  );
}

function NodeIcon({ node, orders }) {
  // Glyph + color per kind
  let glyph = '⭐';
  let fill  = '#7C8DB5';
  if (node.kind === 'pickup') { glyph = '🍔'; fill = '#F0B541'; }
  if (node.kind === 'dropoff') { glyph = '🏠'; fill = '#7CC082'; }
  if (node.kind === 'hub') { glyph = '🏢'; fill = '#7C8DB5'; }

  // For dropoff: highlight if customer has an active order with deadline pressure
  const orderHere = orders?.find((o) => o.dropoff_node_id === node.id && o.status !== 'delivered' && o.status !== 'expired');
  const isUrgent = orderHere && (orderHere.deadline_tick - (orderHere.delivered_tick ?? 0)) < 5;

  return (
    <g className="node-group" transform={`translate(${node.x}, ${node.y})`}>
      <circle r={isUrgent ? 22 : 20} fill={fill} fillOpacity="0.18" stroke={fill} strokeWidth="1.5" />
      <text textAnchor="middle" dy="0.35em" fontSize="20">{glyph}</text>
      <text textAnchor="middle" y="34" fontSize="11" fill="#a4afc7" className="node-label">{node.label}</text>
    </g>
  );
}

function CourierIcon({ courier, x, y, orders }) {
  const carrying = courier.load && orders.find((o) => o.id === courier.load);
  const carryColor = carrying ? STATUS_COLORS.picked : '#5DA9F0';
  const isMoving = courier.status !== 'idle' && courier.target_node_id;

  return (
    <g className={`courier-group${isMoving ? ' moving' : ''}`}
       transform={`translate(${x}, ${y})`}
       style={{ transition: 'transform 0.6s ease-out' }}>
      {/* glow ring when active */}
      {isMoving && <circle r="18" fill={carryColor} fillOpacity="0.2">
        <animate attributeName="r" values="14;20;14" dur="1.5s" repeatCount="indefinite" />
      </circle>}
      <circle r="13" fill={carryColor} stroke="#0f1419" strokeWidth="2" />
      <text textAnchor="middle" dy="0.35em" fontSize="14">🛵</text>
      <text textAnchor="middle" y="-19" fontSize="10" fill="#dde2ed" fontWeight="600">{courier.id.replace('courier_', 'C')}</text>
      {courier.eta_remaining > 0 && (
        <text textAnchor="middle" y="26" fontSize="10" fill="#a4afc7">eta {courier.eta_remaining}</text>
      )}
    </g>
  );
}

// ───────────────────────────────────────────────────────────────
// Orders panel
// ───────────────────────────────────────────────────────────────

export function OrdersPanel({ snapshot, maxTicks }) {
  if (!snapshot) return null;
  const orders = [...snapshot.orders].sort((a, b) => {
    // sort: in-progress first, then by deadline
    const order = { ready: 0, queued: 1, picked: 2, delivered: 3, expired: 4 };
    return (order[a.status] - order[b.status]) || (a.deadline_tick - b.deadline_tick);
  });

  return (
    <div className="panel orders-panel">
      <div className="panel-header">orders<span className="panel-meta">{orders.filter(o => o.status === 'delivered').length}/{orders.length} delivered</span></div>
      <div className="orders-list">
        {orders.map((o) => <OrderCard key={o.id} order={o} tick={snapshot.tick} maxTicks={maxTicks} />)}
      </div>
    </div>
  );
}

function OrderCard({ order, tick, maxTicks }) {
  const ticksLeft = order.deadline_tick - tick;
  const isUrgent = order.status !== 'delivered' && order.status !== 'expired' && ticksLeft <= 3;
  return (
    <div className={`order-card status-${order.status}${isUrgent ? ' urgent' : ''}`}>
      <div className="order-row1">
        <span className="order-id">{order.id}</span>
        <span className={`order-kind kind-${order.kind}`}>{order.kind}</span>
        <span className="order-status">{order.status}</span>
      </div>
      <div className="order-row2">
        <span>{order.pickup_node_id} → {order.dropoff_node_id}</span>
      </div>
      <div className="order-row3">
        {order.status === 'delivered' ? (
          <span className="order-delivered-info">✓ at t{order.delivered_tick}</span>
        ) : order.status === 'expired' ? (
          <span className="order-delivered-info">✗ expired</span>
        ) : (
          <span className={isUrgent ? 'deadline-urgent' : ''}>⏱ {ticksLeft}t left</span>
        )}
        <span className="order-assigned">
          {order.assigned_courier_id ? `→ ${order.assigned_courier_id.replace('courier_', 'C')}` : '—'}
        </span>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────────
// Agent feed
// ───────────────────────────────────────────────────────────────

export function AgentFeed({ trajectory, currentTick, onJumpToTick }) {
  return (
    <div className="panel agent-feed">
      <div className="panel-header">agent feed</div>
      <div className="feed-list">
        {trajectory.slice(0, currentTick + 1).reverse().map((step, idx) => (
          <FeedRow key={step.tick}
                   step={step}
                   isCurrent={step.tick === currentTick}
                   onClick={() => onJumpToTick(step.tick)} />
        ))}
      </div>
    </div>
  );
}

function FeedRow({ step, isCurrent, onClick }) {
  const a = step.action;
  let label, args = '';
  if (!a) { label = '(reset)'; }
  else if (a.type === 'assign') { label = 'assign'; args = `${a.courier_id} ← ${a.order_id}`; }
  else if (a.type === 'reposition') { label = 'reposition'; args = `${a.courier_id} → ${a.node_id}`; }
  else if (a.type === 'hold') { label = 'hold'; }
  else if (a.type === 'prioritize') { label = 'prioritize'; args = a.order_id; }
  else if (a.type === 'view_dashboard') { label = 'view_dashboard'; }
  else if (a.type === 'finish_shift') { label = 'finish_shift'; }
  else { label = a.type || 'unknown'; }

  return (
    <div className={`feed-row${isCurrent ? ' current' : ''}`} onClick={onClick}>
      <span className="feed-tick">t{step.tick}</span>
      <span className="feed-action">{label}</span>
      <span className="feed-args">{args}</span>
      {step.events && step.events.length > 0 && (
        <span className="feed-events">{step.events[0]}</span>
      )}
    </div>
  );
}

// ───────────────────────────────────────────────────────────────
// Reward panel
// ───────────────────────────────────────────────────────────────

const REWARD_KEYS = [
  ['total',     'total reward',  '#5DA9F0'],
  ['success',   'delivered',     '#7CC082'],
  ['on_time',   'on-time bonus', '#7CC082'],
  ['progress',  'progress',      '#5DA9F0'],
  ['step_cost', 'step cost',     '#a4afc7'],
  ['idle',      'idle',          '#a4afc7'],
  ['invalid',   'invalid',       '#E07560'],
  ['late',      'late',          '#E07560'],
  ['churn',     'churn',         '#a4afc7'],
  ['fairness',  'fairness',      '#a4afc7'],
];

export function RewardPanel({ snapshot, cumulative }) {
  if (!snapshot) return null;
  const rb = snapshot.reward_breakdown || {};
  // For visualization, compute cumulative for each component (mock has only this-step values).
  // The bar width is proportional to abs(value), and bars are colored by sign.

  const maxAbs = Math.max(0.5, ...REWARD_KEYS.map(([k]) => Math.abs(rb[k] || 0)));

  return (
    <div className="panel reward-panel">
      <div className="panel-header">reward
        <span className="panel-meta cumulative-reward">total: {cumulative >= 0 ? '+' : ''}{cumulative.toFixed(2)}</span>
      </div>
      <div className="reward-rows">
        {REWARD_KEYS.map(([key, label, color]) => {
          const val = rb[key] ?? 0;
          const widthPct = (Math.abs(val) / maxAbs) * 100;
          const isNeg = val < 0;
          return (
            <div className="reward-row" key={key}>
              <span className="reward-label">{label}</span>
              <span className="reward-bar-track">
                <span className={`reward-bar ${isNeg ? 'neg' : 'pos'}`}
                      style={{ width: `${widthPct}%`, background: color, opacity: val === 0 ? 0.15 : 1 }} />
              </span>
              <span className={`reward-val ${isNeg ? 'neg' : ''}`}>
                {val > 0 ? '+' : ''}{val.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────────
// Top control bar
// ───────────────────────────────────────────────────────────────

export function ControlBar({
  scenarios,
  scenarioId,
  onScenarioChange,
  modes,
  modeId,
  onModeChange,
  isPlaying,
  onPlayToggle,
  onStepBack,
  onStepForward,
  onReset,
  speed,
  onSpeedChange,
  comparisonMode,
  onComparisonToggle,
}) {
  return (
    <div className="control-bar">
      <div className="control-bar-left">
        <span className="brand">Dispatch Arena</span>
        <select className="ctrl-select" value={scenarioId} onChange={(e) => onScenarioChange(e.target.value)}>
          {Object.entries(scenarios).map(([id, s]) => (
            <option key={id} value={id}>
              {s.metadata.theme} ({s.metadata.difficulty})
            </option>
          ))}
        </select>
        {!comparisonMode && (
          <select className="ctrl-select" value={modeId} onChange={(e) => onModeChange(e.target.value)}>
            {modes.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        )}
      </div>
      <div className="control-bar-right">
        <button className="ctrl-btn" onClick={onStepBack}>⏮</button>
        <button className="ctrl-btn primary" onClick={onPlayToggle}>{isPlaying ? '⏸' : '▶'}</button>
        <button className="ctrl-btn" onClick={onStepForward}>⏭</button>
        <button className="ctrl-btn" onClick={onReset}>⟲</button>
        <span className="ctrl-divider" />
        {[1, 2, 4].map((s) => (
          <button key={s}
                  className={`ctrl-btn speed${speed === s ? ' active' : ''}`}
                  onClick={() => onSpeedChange(s)}>
            {s}×
          </button>
        ))}
        <span className="ctrl-divider" />
        <button className={`ctrl-btn ${comparisonMode ? 'active' : ''}`}
                onClick={onComparisonToggle}
                title="Compare heuristic vs trained side-by-side">
          ⇆ compare
        </button>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────────
// Tick scrubber (timeline at the bottom)
// ───────────────────────────────────────────────────────────────

export function TickScrubber({ trajectory, currentTick, onTickChange, maxTicks }) {
  const totalLen = trajectory.length;
  return (
    <div className="tick-scrubber">
      <div className="tick-meta">
        <span>tick {currentTick} / {maxTicks}</span>
        <span className="tick-events">{trajectory[currentTick]?.events?.[0] || ''}</span>
      </div>
      <input type="range"
             min={0}
             max={Math.max(0, totalLen - 1)}
             value={currentTick}
             onChange={(e) => onTickChange(parseInt(e.target.value, 10))}
             className="tick-slider" />
    </div>
  );
}

// ───────────────────────────────────────────────────────────────
// Final summary card (shown at end of episode)
// ───────────────────────────────────────────────────────────────

export function FinalCard({ trajectory, scenario }) {
  const last = trajectory[trajectory.length - 1];
  const orders = last.orders;
  const delivered = orders.filter((o) => o.status === 'delivered').length;
  const onTime = orders.filter((o) => o.status === 'delivered' && (o.delivered_tick ?? 99) <= o.deadline_tick).length;
  const expired = orders.filter((o) => o.status === 'expired').length;
  const successRate = (delivered / orders.length) * 100;

  return (
    <div className="final-card">
      <div className="final-header">shift complete · {scenario.metadata.theme}</div>
      <div className="final-stats">
        <Stat label="delivered" value={`${delivered}/${orders.length}`} accent={successRate >= 80 ? 'good' : successRate >= 50 ? 'mid' : 'bad'} />
        <Stat label="on time" value={`${onTime}/${delivered}`} />
        <Stat label="expired" value={expired} accent={expired > 0 ? 'bad' : 'good'} />
        <Stat label="reward" value={(last.cumulative_reward ?? 0).toFixed(1)} accent={last.cumulative_reward > 0 ? 'good' : 'bad'} />
        <Stat label="ticks" value={`${last.tick}/${scenario.metadata.max_ticks}`} />
      </div>
    </div>
  );
}

function Stat({ label, value, accent }) {
  return (
    <div className={`stat${accent ? ' accent-' + accent : ''}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
