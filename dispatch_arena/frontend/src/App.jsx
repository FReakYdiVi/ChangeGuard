// Main app: composes all panels + manages playback state.
//
// State:
//   scenarioId    — which scenario is loaded
//   modeId        — which model's trajectory is being shown ("trained" | "heuristic" | "prompt_only")
//   currentTick   — index into trajectory
//   isPlaying     — autoplay on
//   speed         — 1× / 2× / 4× (lower delay = higher speed)
//   comparison    — show two columns side-by-side (heuristic vs trained)

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { SCENARIOS } from './fixtures.js';
import {
  ScenarioMap, OrdersPanel, AgentFeed, RewardPanel,
  ControlBar, TickScrubber, FinalCard,
} from './components.jsx';
import './styles.css';

const MODES = [
  { id: 'trained',     label: '🤖 trained agent (Qwen3 + GRPO)' },
  { id: 'heuristic',   label: '⚙ greedy heuristic (baseline)' },
  { id: 'prompt_only', label: '💬 untrained LLM (zero-shot)' },
];

const TICK_BASE_MS = 700;  // 1× speed = one tick per 700 ms

function ScenarioSummary({ scenario }) {
  const m = scenario.metadata;
  return (
    <div className="scenario-summary">
      <div className="scenario-row">
        <span className={`difficulty-badge ${m.difficulty}`}>{m.difficulty}</span>
        <span className="scenario-meta">{m.num_couriers} couriers · {m.num_orders} orders · {m.max_ticks} ticks</span>
      </div>
      <div className="scenario-skills">
        {m.skill_focus.map((s) => <span key={s} className="skill-tag">{s.replace(/_/g, ' ')}</span>)}
      </div>
      <div className="scenario-desc">{m.description}</div>
    </div>
  );
}

function PlaybackPanel({ scenarioId, scenario, modeId, currentTick, onJumpToTick }) {
  const trajectory = scenario.trajectories[modeId] || scenario.trajectories.trained;
  const snapshot = trajectory[Math.min(currentTick, trajectory.length - 1)];
  const isFinished = currentTick >= trajectory.length - 1;

  return (
    <div className="playback-col">
      <div className="map-area">
        <ScenarioMap nodes={scenario.nodes} snapshot={snapshot} />
        {isFinished && <div className="map-final-overlay"><FinalCard trajectory={trajectory} scenario={scenario} /></div>}
      </div>
      <OrdersPanel snapshot={snapshot} maxTicks={scenario.metadata.max_ticks} />
      <RewardPanel snapshot={snapshot} cumulative={snapshot.cumulative_reward ?? 0} />
      <AgentFeed trajectory={trajectory} currentTick={currentTick} onJumpToTick={onJumpToTick} />
    </div>
  );
}

function App() {
  const scenarioIds = Object.keys(SCENARIOS);
  const [scenarioId, setScenarioId] = useState(scenarioIds[0]);
  const [modeId, setModeId] = useState('trained');
  const [currentTick, setCurrentTick] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [comparisonMode, setComparisonMode] = useState(false);

  const scenario = SCENARIOS[scenarioId];
  const trajectory = scenario.trajectories[modeId] || scenario.trajectories.trained;
  const trajectoryLen = useMemo(() => {
    if (comparisonMode) {
      return Math.max(scenario.trajectories.trained.length, scenario.trajectories.heuristic.length);
    }
    return trajectory.length;
  }, [scenario, trajectory, comparisonMode]);

  // Reset tick when scenario or mode changes
  useEffect(() => { setCurrentTick(0); setIsPlaying(false); }, [scenarioId, modeId, comparisonMode]);

  // Playback loop
  const intervalRef = useRef(null);
  useEffect(() => {
    if (!isPlaying) {
      if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
      return;
    }
    intervalRef.current = setInterval(() => {
      setCurrentTick((prev) => {
        if (prev >= trajectoryLen - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, TICK_BASE_MS / speed);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isPlaying, speed, trajectoryLen]);

  return (
    <div className="app">
      <ControlBar
        scenarios={SCENARIOS}
        scenarioId={scenarioId}
        onScenarioChange={setScenarioId}
        modes={MODES}
        modeId={modeId}
        onModeChange={setModeId}
        isPlaying={isPlaying}
        onPlayToggle={() => setIsPlaying((p) => !p)}
        onStepBack={() => setCurrentTick((t) => Math.max(0, t - 1))}
        onStepForward={() => setCurrentTick((t) => Math.min(trajectoryLen - 1, t + 1))}
        onReset={() => { setCurrentTick(0); setIsPlaying(false); }}
        speed={speed}
        onSpeedChange={setSpeed}
        comparisonMode={comparisonMode}
        onComparisonToggle={() => setComparisonMode((c) => !c)}
      />

      <ScenarioSummary scenario={scenario} />

      <div className={`workspace${comparisonMode ? ' compare' : ''}`}>
        {comparisonMode ? (
          <>
            <div className="compare-col">
              <div className="compare-label heuristic">⚙ greedy heuristic</div>
              <PlaybackPanel scenarioId={scenarioId} scenario={scenario} modeId="heuristic"
                             currentTick={currentTick} onJumpToTick={setCurrentTick} />
            </div>
            <div className="compare-col">
              <div className="compare-label trained">🤖 trained agent</div>
              <PlaybackPanel scenarioId={scenarioId} scenario={scenario} modeId="trained"
                             currentTick={currentTick} onJumpToTick={setCurrentTick} />
            </div>
          </>
        ) : (
          <PlaybackPanel scenarioId={scenarioId} scenario={scenario} modeId={modeId}
                         currentTick={currentTick} onJumpToTick={setCurrentTick} />
        )}
      </div>

      <TickScrubber trajectory={trajectory}
                    currentTick={currentTick}
                    onTickChange={setCurrentTick}
                    maxTicks={scenario.metadata.max_ticks} />
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
