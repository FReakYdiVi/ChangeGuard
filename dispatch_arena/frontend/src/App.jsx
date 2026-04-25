import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const AUTO_STEP_DELAY_MS = 900;

function heuristicAction(observation) {
  const state = observation.state;
  if (state.mode === "normal") {
    const courier = state.couriers.find((candidate) => candidate.status === "idle" && !candidate.load);
    const order = state.orders.find((candidate) => ["queued", "ready"].includes(candidate.status) && !candidate.assigned_courier_id);
    return courier && order
      ? { action_type: "assign", courier_id: courier.id, order_id: order.id }
      : { action_type: "hold" };
  }
  for (const action of ["pickup", "dropoff", "go_pickup", "go_dropoff", "wait"]) {
    if (observation.legal_actions.includes(action)) return action;
  }
  return "wait";
}

function actionFromName(actionName, observation) {
  if (observation.state.mode === "mini") return actionName;
  if (actionName === "assign") return heuristicAction(observation);
  if (actionName === "prioritize") {
    const order = observation.state.orders.find((candidate) => ["queued", "ready"].includes(candidate.status));
    return { action_type: "prioritize", order_id: order?.id };
  }
  if (actionName === "reposition") {
    const courier = observation.state.couriers.find((candidate) => candidate.status === "idle" && !candidate.load);
    const target = observation.state.nodes.find((node) => node.id !== courier?.node_id);
    return { action_type: "reposition", courier_id: courier?.id, node_id: target?.id };
  }
  return { action_type: "hold" };
}

function formatAction(action) {
  if (!action) return "Waiting for the next decision";
  if (typeof action === "string") return action;
  const parts = [action.action_type];
  if (action.courier_id) parts.push(action.courier_id);
  if (action.order_id) parts.push(action.order_id);
  if (action.node_id) parts.push(action.node_id);
  return parts.join(" -> ");
}

function demoConfig(mode) {
  return mode === "normal" ? { mode, num_couriers: 3, num_orders: 5, max_ticks: 18 } : { mode, max_ticks: 12 };
}

function orderStatusEvent(order, tick) {
  const labels = {
    ready: `Order ${order.id} ready`,
    picked: `Order ${order.id} picked up`,
    delivered: `Order ${order.id} delivered`,
    expired: `Order ${order.id} expired`,
    queued: `Order ${order.id} queued`,
  };
  return {
    type: order.status,
    tick,
    label: labels[order.status] || `Order ${order.id} ${order.status}`,
    detail: `Deadline tick ${order.deadline_tick}`,
  };
}

function buildInitialEvents(observation) {
  const state = observation?.state;
  if (!state) return [];
  const events = state.orders.map((order) => ({
    type: "received",
    tick: state.tick,
    label: `Order ${order.id} received`,
    detail: `${order.pickup_node_id} to ${order.dropoff_node_id}`,
  }));
  state.orders
    .filter((order) => order.status !== "queued")
    .forEach((order) => events.push(orderStatusEvent(order, state.tick)));
  return events.filter(Boolean);
}

function buildTransitionEvents(previous, next, action, source) {
  const state = next?.state;
  if (!state) return [];
  const events = [];
  if (action) {
    events.push({
      type: source === "model" ? "model" : "manual",
      tick: state.tick,
      label: source === "model" ? "Policy action selected" : "Manual action selected",
      detail: formatAction(action),
    });
  }

  const previousOrders = new Map((previous?.state?.orders || []).map((order) => [order.id, order]));
  state.orders.forEach((order) => {
    const oldOrder = previousOrders.get(order.id);
    if (!oldOrder) {
      events.push({
        type: "received",
        tick: state.tick,
        label: `Order ${order.id} received`,
        detail: `${order.pickup_node_id} to ${order.dropoff_node_id}`,
      });
      return;
    }
    if (oldOrder.status !== order.status) events.push(orderStatusEvent(order, state.tick));
    if (!oldOrder.assigned_courier_id && order.assigned_courier_id) {
      events.push({
        type: "assigned",
        tick: state.tick,
        label: `Courier ${order.assigned_courier_id} assigned`,
        detail: `Order ${order.id}`,
      });
    }
  });

  (next.info?.events || []).forEach((event) => {
    events.push({ type: "simulator", tick: state.tick, label: event, detail: "Simulator event" });
  });
  return events.filter(Boolean);
}

function stampEvents(events) {
  return events.map((event, index) => ({
    ...event,
    id: `${Date.now()}-${index}-${event.type}-${event.label}`,
  }));
}

function App() {
  const [mode, setMode] = useState("mini");
  const [seed, setSeed] = useState(7);
  const [sessionId, setSessionId] = useState(null);
  const [observation, setObservation] = useState(null);
  const [replay, setReplay] = useState([]);
  const [eventFeed, setEventFeed] = useState([]);
  const [lastModelAction, setLastModelAction] = useState(null);
  const [isAutoRunning, setIsAutoRunning] = useState(false);

  async function post(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  async function startSession() {
    const config = demoConfig(mode);
    const data = await post("/api/sessions", { mode, seed: Number(seed), config });
    setSessionId(data.session_id);
    setObservation(data.observation);
    setReplay([]);
    setLastModelAction(null);
    setEventFeed(stampEvents(buildInitialEvents(data.observation)));
    return data;
  }

  async function step(action, source = "manual", targetSessionId = sessionId, previous = observation) {
    if (!targetSessionId || previous?.done) return undefined;
    const data = await post(`/api/sessions/${targetSessionId}/step`, { action });
    setObservation(data.observation);
    setEventFeed((current) => [
      ...stampEvents(buildTransitionEvents(previous, data.observation, action, source)),
      ...current,
    ].slice(0, 32));
    return data;
  }

  async function stepModel() {
    let current = observation;
    let targetSessionId = sessionId;
    if (!current || current.done) {
      const data = await startSession();
      current = data.observation;
      targetSessionId = data.session_id;
    }
    const action = heuristicAction(current);
    setLastModelAction(action);
    await step(action, "model", targetSessionId, current);
  }

  async function startAutoRun() {
    if (!observation || observation.done) await startSession();
    setIsAutoRunning(true);
  }

  useEffect(() => {
    if (!isAutoRunning || !sessionId || !observation) return undefined;
    if (observation.done) {
      setIsAutoRunning(false);
      return undefined;
    }
    const timer = window.setTimeout(() => {
      stepModel();
    }, AUTO_STEP_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [isAutoRunning, observation, sessionId]);

  useEffect(() => {
    if (!sessionId) return;
    fetch(`/api/sessions/${sessionId}/replay`).then((res) => res.json()).then((data) => setReplay(data.records));
  }, [sessionId, observation]);

  const state = observation?.state;
  return (
    <main>
      <header className="hero">
        <div>
          <p className="eyebrow">Dispatch Arena demo</p>
          <h1>Order arrives, policy acts, courier moves.</h1>
          <p className="hero-copy">
            The visual policy uses the in-repo heuristic today and can be swapped for a trained model action endpoint later.
          </p>
        </div>
        <ModelActionCard observation={observation} action={lastModelAction} isAutoRunning={isAutoRunning} />
      </header>

      <DemoControls
        mode={mode}
        seed={seed}
        isAutoRunning={isAutoRunning}
        hasObservation={Boolean(observation)}
        isDone={Boolean(observation?.done)}
        onModeChange={(value) => {
          setMode(value);
          setIsAutoRunning(false);
        }}
        onSeedChange={setSeed}
        onStart={startSession}
        onStepModel={stepModel}
        onAutoRun={startAutoRun}
        onPause={() => setIsAutoRunning(false)}
      />

      {state && (
        <section className="demo-layout">
          <Panel title="Dispatch Board" wide>
            <DispatchBoard state={state} lastAction={lastModelAction} />
          </Panel>
          <Panel title="Order Feed">
            <OrderFeed events={eventFeed} />
          </Panel>
          <Panel title="Manual Actions">
            <div className="button-row">
              {observation.legal_actions.map((actionName) => (
                <button key={actionName} onClick={() => step(actionFromName(actionName, observation), "manual")}>
                  {actionName}
                </button>
              ))}
            </div>
          </Panel>
          <Panel title="Couriers">
            <CourierTable couriers={state.couriers} />
          </Panel>
          <Panel title="Orders">
            <OrderTable orders={state.orders} />
          </Panel>
          <Panel title="Reward">
            <MetricGrid observation={observation} replayCount={replay.length} />
          </Panel>
        </section>
      )}
    </main>
  );
}

function DemoControls({
  mode,
  seed,
  isAutoRunning,
  hasObservation,
  isDone,
  onModeChange,
  onSeedChange,
  onStart,
  onStepModel,
  onAutoRun,
  onPause,
}) {
  return (
    <section className="controls">
      <label>
        Mode
        <select value={mode} onChange={(event) => onModeChange(event.target.value)}>
          <option value="mini">mini</option>
          <option value="normal">normal</option>
        </select>
      </label>
      <label>
        Seed
        <input type="number" value={seed} onChange={(event) => onSeedChange(event.target.value)} />
      </label>
      <button onClick={onStart}>Start Demo</button>
      <button onClick={onStepModel} disabled={hasObservation && isDone}>Step Model</button>
      {isAutoRunning ? (
        <button onClick={onPause}>Pause</button>
      ) : (
        <button onClick={onAutoRun} disabled={hasObservation && isDone}>Auto Run</button>
      )}
    </section>
  );
}

function ModelActionCard({ observation, action, isAutoRunning }) {
  const status = observation?.done ? observation.verifier_status : isAutoRunning ? "auto running" : "ready";
  return (
    <aside className="model-card">
      <span className="status-pill">{status}</span>
      <h2>Model Action</h2>
      <p className="action-text">{formatAction(action)}</p>
      <small>Source: demo heuristic policy</small>
    </aside>
  );
}

function DispatchBoard({ state, lastAction }) {
  const positions = useMemo(() => nodePositions(state.nodes), [state.nodes]);
  const links = useMemo(() => graphLinks(state), [state]);
  return (
    <div className="board-wrap">
      <svg className="dispatch-board" viewBox="0 0 720 360" role="img" aria-label="Dispatch board">
        {links.map((link) => (
          <line key={link.key} x1={positions[link.from].x} y1={positions[link.from].y} x2={positions[link.to].x} y2={positions[link.to].y} />
        ))}
        {state.nodes.map((node) => (
          <g key={node.id} className="board-node" transform={`translate(${positions[node.id].x}, ${positions[node.id].y})`}>
            <circle r="38" />
            <text y="-3">{node.label}</text>
            <text y="15" className="muted-text">{node.id}</text>
          </g>
        ))}
        {state.orders.map((order, index) => {
          const pickup = positions[order.pickup_node_id] || positions[state.nodes[0]?.id];
          const dropoff = positions[order.dropoff_node_id] || pickup;
          const atDropoff = ["delivered", "expired"].includes(order.status);
          const x = (atDropoff ? dropoff.x : pickup.x) + 42;
          const y = (atDropoff ? dropoff.y : pickup.y) - 30 + (index % 3) * 24;
          return (
            <g key={order.id} className={`order-marker ${order.status}`} transform={`translate(${x}, ${y})`}>
              <rect x="-28" y="-13" width="56" height="26" rx="13" />
              <text y="5">{order.id}</text>
            </g>
          );
        })}
        {state.couriers.map((courier, index) => {
          const position = positions[courier.node_id] || positions[state.nodes[0]?.id];
          return (
            <g
              key={courier.id}
              className={`courier-marker ${courier.status}`}
              transform={`translate(${position.x - 44 + index * 22}, ${position.y + 52})`}
            >
              <circle r="15" />
              <text y="5">{courier.id.replace("courier_", "c")}</text>
            </g>
          );
        })}
      </svg>
      <div className="board-caption">
        <span>Tick {state.tick} / {state.max_ticks}</span>
        <span>Last policy action: {formatAction(lastAction)}</span>
      </div>
    </div>
  );
}

function nodePositions(nodes) {
  const knownMini = {
    hub: { x: 95, y: 180 },
    pickup: { x: 360, y: 95 },
    dropoff: { x: 625, y: 230 },
  };
  if (nodes.length <= 3 && nodes.every((node) => knownMini[node.id])) {
    return Object.fromEntries(nodes.map((node) => [node.id, knownMini[node.id]]));
  }
  const center = { x: 360, y: 180 };
  const radius = 125;
  return Object.fromEntries(nodes.map((node, index) => {
    const angle = (2 * Math.PI * index) / Math.max(nodes.length, 1) - Math.PI / 2;
    return [node.id, { x: center.x + radius * Math.cos(angle), y: center.y + radius * Math.sin(angle) }];
  }));
}

function graphLinks(state) {
  const seen = new Set();
  const links = [];
  Object.entries(state.travel_time_matrix || {}).forEach(([from, destinations]) => {
    Object.keys(destinations || {}).forEach((to) => {
      if (from === to) return;
      const key = [from, to].sort().join(":");
      if (seen.has(key)) return;
      seen.add(key);
      links.push({ key, from, to });
    });
  });
  return links.filter((link) => state.nodes.some((node) => node.id === link.from) && state.nodes.some((node) => node.id === link.to));
}

function OrderFeed({ events }) {
  if (!events.length) return <p className="empty-state">Start a demo to see order events.</p>;
  return (
    <ol className="event-feed">
      {events.slice(0, 10).map((event) => (
        <li key={event.id} className={event.type}>
          <span>t{event.tick}</span>
          <strong>{event.label}</strong>
          <small>{event.detail}</small>
        </li>
      ))}
    </ol>
  );
}

function CourierTable({ couriers }) {
  return (
    <table>
      <tbody>
        {couriers.map((courier) => (
          <tr key={courier.id}>
            <td>{courier.id}</td>
            <td>{courier.node_id}</td>
            <td><span className="badge">{courier.status}</span></td>
            <td>{courier.load || courier.assigned_order_id || "free"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function OrderTable({ orders }) {
  return (
    <table>
      <tbody>
        {orders.map((order) => (
          <tr key={order.id}>
            <td>{order.id}</td>
            <td><span className={`badge ${order.status}`}>{order.status}</span></td>
            <td>{order.pickup_node_id} to {order.dropoff_node_id}</td>
            <td>due t{order.deadline_tick}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function MetricGrid({ observation, replayCount }) {
  return (
    <div className="metrics">
      <div><span>Reward</span><strong>{observation.reward.toFixed(2)}</strong></div>
      <div><span>Total</span><strong>{observation.state.total_reward.toFixed(2)}</strong></div>
      <div><span>Backlog</span><strong>{observation.state.backlog}</strong></div>
      <div><span>Replay</span><strong>{replayCount}</strong></div>
    </div>
  );
}

function Panel({ title, children, wide = false }) {
  return <div className={`card ${wide ? "wide-card" : ""}`}><h2>{title}</h2>{children}</div>;
}

createRoot(document.getElementById("root")).render(<App />);
