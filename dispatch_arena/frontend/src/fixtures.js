// Mock replay fixtures: realistic per-tick trajectories that the real backend
// will produce. Two scenarios so the picker has something to switch between.
//
// Each fixture has:
//   metadata: how the scenario is described in the catalog
//   nodes:    {id, kind, label, x, y} laid out in a 900x520 viewBox
//   trajectory: array of per-tick snapshots
//     {tick, action, couriers[], orders[], reward_breakdown, events[]}
//
// Coords are stylized — hub centered, stores on inner ring, customers on outer
// ring. The real backend will eventually pre-compute these from real episodes.

const PALETTE = {
  hub: '#7C8DB5',
  store: '#F0B541',
  customer: '#7CC082',
  courier: '#5DA9F0',
};

// ───────────────────────────────────────────────────────────────
// Scenario 1: easy — Office Park Predictable (2 couriers, 3 orders)
// ───────────────────────────────────────────────────────────────

const easyNodes = [
  { id: 'hub',         kind: 'hub',      label: 'Hub',          x: 450, y: 260 },
  { id: 'store_0',     kind: 'pickup',   label: 'Empire Eats',  x: 270, y: 180 },
  { id: 'store_1',     kind: 'pickup',   label: 'Burger Bay',   x: 630, y: 180 },
  { id: 'customer_0',  kind: 'dropoff',  label: 'Tower A',      x: 160, y: 360 },
  { id: 'customer_1',  kind: 'dropoff',  label: 'Tower B',      x: 450, y: 460 },
  { id: 'customer_2',  kind: 'dropoff',  label: 'Tower C',      x: 740, y: 360 },
];

const easyTrajectory = [
  // tick 0 — initial state
  {
    tick: 0,
    action: null,
    couriers: [
      { id: 'courier_0', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_1', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'queued', deadline_tick: 14, assigned_courier_id: null },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_2', status: 'queued', deadline_tick: 14, assigned_courier_id: null },
      { id: 'order_2', kind: 'grocery', pickup_node_id: 'store_0', dropoff_node_id: 'customer_1', status: 'queued', deadline_tick: 16, assigned_courier_id: null },
    ],
    reward_breakdown: { step_cost: 0, progress: 0, success: 0, on_time: 0, late: 0, invalid: 0, idle: 0, churn: 0, fairness: 0, total: 0 },
    cumulative_reward: 0,
    events: ['shift started — 3 orders, 2 couriers'],
  },
  {
    tick: 1,
    action: { type: 'assign', courier_id: 'courier_0', order_id: 'order_0' },
    couriers: [
      { id: 'courier_0', node_id: 'hub',     status: 'to_pickup', load: null, target_node_id: 'store_0', eta_remaining: 2 },
      { id: 'courier_1', node_id: 'hub',     status: 'idle',      load: null, target_node_id: null, eta_remaining: 0 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'queued', deadline_tick: 14, assigned_courier_id: 'courier_0' },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_2', status: 'queued', deadline_tick: 14, assigned_courier_id: null },
      { id: 'order_2', kind: 'grocery', pickup_node_id: 'store_0', dropoff_node_id: 'customer_1', status: 'queued', deadline_tick: 16, assigned_courier_id: null },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0.5, success: 0, on_time: 0, late: 0, invalid: 0, idle: -0.05, churn: 0, fairness: 0, total: 0.35 },
    cumulative_reward: 0.35,
    events: ['courier_0 → store_0 (eta 2)'],
  },
  {
    tick: 2,
    action: { type: 'assign', courier_id: 'courier_1', order_id: 'order_1' },
    couriers: [
      { id: 'courier_0', node_id: 'hub',     status: 'to_pickup', load: null, target_node_id: 'store_0', eta_remaining: 1 },
      { id: 'courier_1', node_id: 'hub',     status: 'to_pickup', load: null, target_node_id: 'store_1', eta_remaining: 2 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'ready',  deadline_tick: 14, assigned_courier_id: 'courier_0' },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_2', status: 'queued', deadline_tick: 14, assigned_courier_id: 'courier_1' },
      { id: 'order_2', kind: 'grocery', pickup_node_id: 'store_0', dropoff_node_id: 'customer_1', status: 'queued', deadline_tick: 16, assigned_courier_id: null },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0.5, success: 0, on_time: 0, late: 0, invalid: 0, idle: 0, churn: 0, fairness: 0, total: 0.4 },
    cumulative_reward: 0.75,
    events: ['order_0 ready', 'courier_1 → store_1 (eta 2)'],
  },
  {
    tick: 3,
    action: { type: 'hold' },
    couriers: [
      { id: 'courier_0', node_id: 'store_0', status: 'to_dropoff', load: 'order_0', target_node_id: 'customer_0', eta_remaining: 2 },
      { id: 'courier_1', node_id: 'hub',     status: 'to_pickup',  load: null,      target_node_id: 'store_1', eta_remaining: 1 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'picked', deadline_tick: 14, assigned_courier_id: 'courier_0' },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_2', status: 'ready',  deadline_tick: 14, assigned_courier_id: 'courier_1' },
      { id: 'order_2', kind: 'grocery', pickup_node_id: 'store_0', dropoff_node_id: 'customer_1', status: 'queued', deadline_tick: 16, assigned_courier_id: null },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0.5, success: 0, on_time: 0, late: 0, invalid: 0, idle: 0, churn: 0, fairness: 0, total: 0.4 },
    cumulative_reward: 1.15,
    events: ['courier_0 picked order_0', 'order_1 ready'],
  },
  {
    tick: 4,
    action: { type: 'assign', courier_id: 'courier_0', order_id: 'order_2' },
    // Note: courier_0 is busy delivering, so this would actually be invalid; but for visual demo
    // we model it as a valid reposition-after-dropoff plan. In real env this would be courier_0
    // becoming idle after delivery and being reassigned.
    couriers: [
      { id: 'courier_0', node_id: 'store_0', status: 'to_dropoff', load: 'order_0', target_node_id: 'customer_0', eta_remaining: 1 },
      { id: 'courier_1', node_id: 'store_1', status: 'to_dropoff', load: 'order_1', target_node_id: 'customer_2', eta_remaining: 2 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'picked', deadline_tick: 14, assigned_courier_id: 'courier_0' },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_2', status: 'picked', deadline_tick: 14, assigned_courier_id: 'courier_1' },
      { id: 'order_2', kind: 'grocery', pickup_node_id: 'store_0', dropoff_node_id: 'customer_1', status: 'queued', deadline_tick: 16, assigned_courier_id: null },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0.5, success: 0, on_time: 0, late: 0, invalid: 0, idle: 0, churn: 0, fairness: 0, total: 0.4 },
    cumulative_reward: 1.55,
    events: ['courier_1 picked order_1'],
  },
  {
    tick: 5,
    action: { type: 'hold' },
    couriers: [
      { id: 'courier_0', node_id: 'customer_0', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_1', node_id: 'store_1',    status: 'to_dropoff', load: 'order_1', target_node_id: 'customer_2', eta_remaining: 1 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'delivered', deadline_tick: 14, assigned_courier_id: 'courier_0', delivered_tick: 5 },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_2', status: 'picked',    deadline_tick: 14, assigned_courier_id: 'courier_1' },
      { id: 'order_2', kind: 'grocery', pickup_node_id: 'store_0', dropoff_node_id: 'customer_1', status: 'queued',    deadline_tick: 16, assigned_courier_id: null },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0, success: 10.0, on_time: 2.0, late: 0, invalid: 0, idle: 0, churn: 0, fairness: 0, total: 11.9 },
    cumulative_reward: 13.45,
    events: ['order_0 delivered by courier_0 (on time!)'],
  },
  {
    tick: 6,
    action: { type: 'assign', courier_id: 'courier_0', order_id: 'order_2' },
    couriers: [
      { id: 'courier_0', node_id: 'customer_0', status: 'to_pickup', load: null, target_node_id: 'store_0', eta_remaining: 1 },
      { id: 'courier_1', node_id: 'customer_2', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'delivered', deadline_tick: 14, assigned_courier_id: 'courier_0', delivered_tick: 5 },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_2', status: 'delivered', deadline_tick: 14, assigned_courier_id: 'courier_1', delivered_tick: 6 },
      { id: 'order_2', kind: 'grocery', pickup_node_id: 'store_0', dropoff_node_id: 'customer_1', status: 'queued',    deadline_tick: 16, assigned_courier_id: 'courier_0' },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0.5, success: 10.0, on_time: 2.0, late: 0, invalid: 0, idle: -0.05, churn: 0, fairness: 0, total: 12.35 },
    cumulative_reward: 25.8,
    events: ['order_1 delivered by courier_1 (on time!)', 'courier_0 → store_0 (eta 1)'],
  },
  {
    tick: 7,
    action: { type: 'hold' },
    couriers: [
      { id: 'courier_0', node_id: 'store_0',    status: 'to_dropoff', load: 'order_2', target_node_id: 'customer_1', eta_remaining: 2 },
      { id: 'courier_1', node_id: 'customer_2', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'delivered', deadline_tick: 14, assigned_courier_id: 'courier_0', delivered_tick: 5 },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_2', status: 'delivered', deadline_tick: 14, assigned_courier_id: 'courier_1', delivered_tick: 6 },
      { id: 'order_2', kind: 'grocery', pickup_node_id: 'store_0', dropoff_node_id: 'customer_1', status: 'picked',    deadline_tick: 16, assigned_courier_id: 'courier_0' },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0.5, success: 0, on_time: 0, late: 0, invalid: 0, idle: -0.05, churn: 0, fairness: 0, total: 0.35 },
    cumulative_reward: 26.15,
    events: ['courier_0 picked order_2'],
  },
  {
    tick: 8,
    action: { type: 'finish_shift' },
    couriers: [
      { id: 'courier_0', node_id: 'customer_1', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_1', node_id: 'customer_2', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'delivered', deadline_tick: 14, assigned_courier_id: 'courier_0', delivered_tick: 5 },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_2', status: 'delivered', deadline_tick: 14, assigned_courier_id: 'courier_1', delivered_tick: 6 },
      { id: 'order_2', kind: 'grocery', pickup_node_id: 'store_0', dropoff_node_id: 'customer_1', status: 'delivered', deadline_tick: 16, assigned_courier_id: 'courier_0', delivered_tick: 8 },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0, success: 10.0, on_time: 2.0, late: 0, invalid: 0, idle: 0, churn: 0, fairness: 0, total: 11.9 },
    cumulative_reward: 38.05,
    events: ['order_2 delivered by courier_0 (on time!)', 'all 3 orders delivered — shift complete'],
  },
];

// ───────────────────────────────────────────────────────────────
// Scenario 2: hard — Holiday Eve Chaos (5 couriers, 9 orders, traffic + rolling)
// ───────────────────────────────────────────────────────────────

const hardNodes = [
  { id: 'hub',          kind: 'hub',     label: 'Hub',         x: 450, y: 260 },
  { id: 'store_0',      kind: 'pickup',  label: 'BiryaniBros', x: 200, y: 130 },
  { id: 'store_1',      kind: 'pickup',  label: 'PizzaCo',     x: 450, y: 80  },
  { id: 'store_2',      kind: 'pickup',  label: 'NoodleHaus',  x: 700, y: 130 },
  { id: 'store_3',      kind: 'pickup',  label: 'GroceryMart', x: 700, y: 410 },
  { id: 'customer_0',   kind: 'dropoff', label: 'Indiranagar', x: 90,  y: 290 },
  { id: 'customer_1',   kind: 'dropoff', label: 'Koramangala', x: 200, y: 460 },
  { id: 'customer_2',   kind: 'dropoff', label: 'JP Nagar',    x: 450, y: 470 },
  { id: 'customer_3',   kind: 'dropoff', label: 'HSR Layout',  x: 700, y: 470 },
  { id: 'customer_4',   kind: 'dropoff', label: 'Whitefield',  x: 820, y: 290 },
];

const hardTrajectory = [
  {
    tick: 0,
    action: null,
    couriers: [
      { id: 'courier_0', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_1', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_2', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_3', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_4', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'queued', deadline_tick: 9,  assigned_courier_id: null },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_1', status: 'queued', deadline_tick: 11, assigned_courier_id: null },
      { id: 'order_2', kind: 'food',    pickup_node_id: 'store_2', dropoff_node_id: 'customer_2', status: 'queued', deadline_tick: 12, assigned_courier_id: null },
    ],
    reward_breakdown: { step_cost: 0, progress: 0, success: 0, on_time: 0, late: 0, invalid: 0, idle: 0, churn: 0, fairness: 0, total: 0 },
    cumulative_reward: 0,
    events: ['holiday eve shift started — 9 orders queued + 6 rolling, traffic noise active'],
  },
  {
    tick: 1,
    action: { type: 'assign', courier_id: 'courier_0', order_id: 'order_0' },
    couriers: [
      { id: 'courier_0', node_id: 'hub', status: 'to_pickup', load: null, target_node_id: 'store_0', eta_remaining: 3 }, // traffic +1
      { id: 'courier_1', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_2', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_3', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_4', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'ready',  deadline_tick: 9,  assigned_courier_id: 'courier_0' },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_1', status: 'queued', deadline_tick: 11, assigned_courier_id: null },
      { id: 'order_2', kind: 'food',    pickup_node_id: 'store_2', dropoff_node_id: 'customer_2', status: 'queued', deadline_tick: 12, assigned_courier_id: null },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0.5, success: 0, on_time: 0, late: 0, invalid: 0, idle: -0.2, churn: 0, fairness: 0, total: 0.2 },
    cumulative_reward: 0.2,
    events: ['courier_0 → store_0 (eta 3, traffic +1)'],
  },
  {
    tick: 2,
    action: { type: 'assign', courier_id: 'courier_1', order_id: 'order_1' },
    couriers: [
      { id: 'courier_0', node_id: 'hub',     status: 'to_pickup', load: null, target_node_id: 'store_0', eta_remaining: 2 },
      { id: 'courier_1', node_id: 'hub',     status: 'to_pickup', load: null, target_node_id: 'store_1', eta_remaining: 2 },
      { id: 'courier_2', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_3', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_4', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'ready',  deadline_tick: 9,  assigned_courier_id: 'courier_0' },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_1', status: 'ready',  deadline_tick: 11, assigned_courier_id: 'courier_1' },
      { id: 'order_2', kind: 'food',    pickup_node_id: 'store_2', dropoff_node_id: 'customer_2', status: 'queued', deadline_tick: 12, assigned_courier_id: null },
      { id: 'order_3', kind: 'grocery', pickup_node_id: 'store_3', dropoff_node_id: 'customer_3', status: 'queued', deadline_tick: 13, assigned_courier_id: null },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0.5, success: 0, on_time: 0, late: 0, invalid: 0, idle: -0.15, churn: 0, fairness: 0, total: 0.25 },
    cumulative_reward: 0.45,
    events: ['order_1 ready', 'order_3 arrived', 'courier_1 → store_1'],
  },
  {
    tick: 3,
    action: { type: 'assign', courier_id: 'courier_2', order_id: 'order_2' },
    couriers: [
      { id: 'courier_0', node_id: 'hub',     status: 'to_pickup', load: null, target_node_id: 'store_0', eta_remaining: 1 },
      { id: 'courier_1', node_id: 'hub',     status: 'to_pickup', load: null, target_node_id: 'store_1', eta_remaining: 1 },
      { id: 'courier_2', node_id: 'hub',     status: 'to_pickup', load: null, target_node_id: 'store_2', eta_remaining: 3 },
      { id: 'courier_3', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_4', node_id: 'hub', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'ready',  deadline_tick: 9,  assigned_courier_id: 'courier_0' },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_1', status: 'ready',  deadline_tick: 11, assigned_courier_id: 'courier_1' },
      { id: 'order_2', kind: 'food',    pickup_node_id: 'store_2', dropoff_node_id: 'customer_2', status: 'ready',  deadline_tick: 12, assigned_courier_id: 'courier_2' },
      { id: 'order_3', kind: 'grocery', pickup_node_id: 'store_3', dropoff_node_id: 'customer_3', status: 'queued', deadline_tick: 13, assigned_courier_id: null },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0.5, success: 0, on_time: 0, late: 0, invalid: 0, idle: -0.1, churn: 0, fairness: 0, total: 0.3 },
    cumulative_reward: 0.75,
    events: ['order_2 ready', 'courier_2 → store_2 (eta 3, heavy traffic)'],
  },
  {
    tick: 4,
    action: { type: 'reposition', courier_id: 'courier_3', node_id: 'store_3' },
    couriers: [
      { id: 'courier_0', node_id: 'store_0',  status: 'to_dropoff', load: 'order_0', target_node_id: 'customer_0', eta_remaining: 2 },
      { id: 'courier_1', node_id: 'store_1',  status: 'to_dropoff', load: 'order_1', target_node_id: 'customer_1', eta_remaining: 3 },
      { id: 'courier_2', node_id: 'hub',      status: 'to_pickup',  load: null, target_node_id: 'store_2', eta_remaining: 2 },
      { id: 'courier_3', node_id: 'hub',      status: 'repositioning', load: null, target_node_id: 'store_3', eta_remaining: 2 },
      { id: 'courier_4', node_id: 'hub',      status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'picked', deadline_tick: 9,  assigned_courier_id: 'courier_0' },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_1', status: 'picked', deadline_tick: 11, assigned_courier_id: 'courier_1' },
      { id: 'order_2', kind: 'food',    pickup_node_id: 'store_2', dropoff_node_id: 'customer_2', status: 'ready',  deadline_tick: 12, assigned_courier_id: 'courier_2' },
      { id: 'order_3', kind: 'grocery', pickup_node_id: 'store_3', dropoff_node_id: 'customer_3', status: 'queued', deadline_tick: 13, assigned_courier_id: null },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0.2, success: 0, on_time: 0, late: 0, invalid: 0, idle: -0.05, churn: 0, fairness: 0, total: 0.05 },
    cumulative_reward: 0.8,
    events: ['courier_0 picked order_0', 'courier_1 picked order_1', 'courier_3 → store_3 (pre-staging)'],
  },
  {
    tick: 5,
    action: { type: 'assign', courier_id: 'courier_4', order_id: 'order_3' },
    couriers: [
      { id: 'courier_0', node_id: 'store_0',  status: 'to_dropoff', load: 'order_0', target_node_id: 'customer_0', eta_remaining: 1 },
      { id: 'courier_1', node_id: 'store_1',  status: 'to_dropoff', load: 'order_1', target_node_id: 'customer_1', eta_remaining: 2 },
      { id: 'courier_2', node_id: 'hub',      status: 'to_pickup',  load: null, target_node_id: 'store_2', eta_remaining: 1 },
      { id: 'courier_3', node_id: 'store_3',  status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_4', node_id: 'hub',      status: 'to_pickup',  load: null, target_node_id: 'store_3', eta_remaining: 3 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'picked', deadline_tick: 9,  assigned_courier_id: 'courier_0' },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_1', status: 'picked', deadline_tick: 11, assigned_courier_id: 'courier_1' },
      { id: 'order_2', kind: 'food',    pickup_node_id: 'store_2', dropoff_node_id: 'customer_2', status: 'ready',  deadline_tick: 12, assigned_courier_id: 'courier_2' },
      { id: 'order_3', kind: 'grocery', pickup_node_id: 'store_3', dropoff_node_id: 'customer_3', status: 'ready',  deadline_tick: 13, assigned_courier_id: 'courier_4' },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0.5, success: 0, on_time: 0, late: 0, invalid: 0, idle: 0, churn: 0, fairness: 0, total: 0.4 },
    cumulative_reward: 1.2,
    events: ['courier_3 finished reposition', 'order_3 ready'],
  },
  {
    tick: 6,
    action: { type: 'hold' },
    couriers: [
      { id: 'courier_0', node_id: 'customer_0', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_1', node_id: 'store_1',    status: 'to_dropoff', load: 'order_1', target_node_id: 'customer_1', eta_remaining: 1 },
      { id: 'courier_2', node_id: 'store_2',    status: 'to_dropoff', load: 'order_2', target_node_id: 'customer_2', eta_remaining: 4 },
      { id: 'courier_3', node_id: 'store_3',    status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_4', node_id: 'hub',        status: 'to_pickup', load: null, target_node_id: 'store_3', eta_remaining: 2 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'delivered', deadline_tick: 9,  assigned_courier_id: 'courier_0', delivered_tick: 6 },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_1', status: 'picked',    deadline_tick: 11, assigned_courier_id: 'courier_1' },
      { id: 'order_2', kind: 'food',    pickup_node_id: 'store_2', dropoff_node_id: 'customer_2', status: 'picked',    deadline_tick: 12, assigned_courier_id: 'courier_2' },
      { id: 'order_3', kind: 'grocery', pickup_node_id: 'store_3', dropoff_node_id: 'customer_3', status: 'ready',     deadline_tick: 13, assigned_courier_id: 'courier_4' },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0, success: 10.0, on_time: 2.0, late: 0, invalid: 0, idle: -0.1, churn: 0, fairness: 0, total: 11.8 },
    cumulative_reward: 13.0,
    events: ['order_0 delivered (on time!)', 'courier_2 picked order_2'],
  },
  {
    tick: 7,
    action: { type: 'hold' },
    couriers: [
      { id: 'courier_0', node_id: 'customer_0', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_1', node_id: 'customer_1', status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_2', node_id: 'store_2',    status: 'to_dropoff', load: 'order_2', target_node_id: 'customer_2', eta_remaining: 3 },
      { id: 'courier_3', node_id: 'store_3',    status: 'idle', load: null, target_node_id: null, eta_remaining: 0 },
      { id: 'courier_4', node_id: 'hub',        status: 'to_pickup',  load: null, target_node_id: 'store_3', eta_remaining: 1 },
    ],
    orders: [
      { id: 'order_0', kind: 'food',    pickup_node_id: 'store_0', dropoff_node_id: 'customer_0', status: 'delivered', deadline_tick: 9,  assigned_courier_id: 'courier_0', delivered_tick: 6 },
      { id: 'order_1', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_1', status: 'delivered', deadline_tick: 11, assigned_courier_id: 'courier_1', delivered_tick: 7 },
      { id: 'order_2', kind: 'food',    pickup_node_id: 'store_2', dropoff_node_id: 'customer_2', status: 'picked',    deadline_tick: 12, assigned_courier_id: 'courier_2' },
      { id: 'order_3', kind: 'grocery', pickup_node_id: 'store_3', dropoff_node_id: 'customer_3', status: 'ready',     deadline_tick: 13, assigned_courier_id: 'courier_4' },
      { id: 'order_4', kind: 'food',    pickup_node_id: 'store_1', dropoff_node_id: 'customer_4', status: 'queued',    deadline_tick: 14, assigned_courier_id: null },
    ],
    reward_breakdown: { step_cost: -0.1, progress: 0, success: 10.0, on_time: 2.0, late: 0, invalid: 0, idle: -0.1, churn: 0, fairness: 0, total: 11.8 },
    cumulative_reward: 24.8,
    events: ['order_1 delivered (on time!)', 'order_4 arrived'],
  },
];

// ───────────────────────────────────────────────────────────────
// Export
// ───────────────────────────────────────────────────────────────

export const SCENARIOS = {
  office_park_predictable: {
    metadata: {
      name: 'office_park_predictable_seed7423',
      theme: 'Office park, predictable cluster',
      description: 'Tight cluster of customers near one store. Travel is short, prep is the only meaningful uncertainty.',
      difficulty: 'easy',
      max_ticks: 18,
      num_couriers: 2,
      num_orders: 3,
      skill_focus: ['prep_uncertainty', 'courier_load_balance'],
    },
    nodes: easyNodes,
    trajectories: {
      trained:    easyTrajectory,
      heuristic:  easyTrajectory,        // same in mock; visually identical for easy
      prompt_only: easyTrajectory.slice(0, 6),  // truncated -> only delivers 1 order
    },
  },
  holiday_eve_chaos: {
    metadata: {
      name: 'holiday_eve_chaos_seed8821',
      theme: 'Holiday eve chaos: long-tail + heavy traffic',
      description: 'Long-tail customer mix, heavy traffic, rolling arrivals across the whole shift, max courier count. The hardest anchor.',
      difficulty: 'hard',
      max_ticks: 16,
      num_couriers: 5,
      num_orders: 9,
      skill_focus: ['traffic_noise', 'long_tail_routing', 'rolling_arrivals'],
    },
    nodes: hardNodes,
    trajectories: {
      trained:    hardTrajectory,
      heuristic:  hardTrajectory.slice(0, 6),  // truncated baseline
      prompt_only: hardTrajectory.slice(0, 4),
    },
  },
};

export const PALETTE_EXPORT = PALETTE;

export const STATUS_COLORS = {
  queued:    '#7C8DB5',
  ready:     '#F0B541',
  picked:    '#5DA9F0',
  delivered: '#7CC082',
  expired:   '#E07560',
};

export const COURIER_STATUS_LABELS = {
  idle:            'idle',
  to_pickup:       '→ pickup',
  waiting_pickup:  'waiting at store',
  to_dropoff:      '→ dropoff',
  repositioning:   '↻ pre-staging',
};
