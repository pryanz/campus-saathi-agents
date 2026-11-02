import React, { useState, useEffect } from 'react';

export default function WhatFirstModal({ isOpen, onClose }) {
  const [arrivalTime, setArrivalTime] = useState('10:30 AM');
  const [hasHeavyLuggage, setHasHeavyLuggage] = useState(true);
  const [category, setCategory] = useState('OBC-NCL');
  const [planData, setPlanData] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchPlan = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/planner/what-first', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          arrival_time: arrivalTime,
          has_heavy_luggage: hasHeavyLuggage,
          category: category
        })
      });
      const data = await res.json();
      setPlanData(data);
    } catch (err) {
      console.error('Failed to fetch plan:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchPlan();
    }
  }, [isOpen, arrivalTime, hasHeavyLuggage, category]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-fadeIn">
      <div className="bg-[#0f172a] border border-slate-800 rounded-2xl w-full max-w-2xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* modal header */}
        <div className="px-6 py-4 bg-[#0b1222] border-b border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-amber-500 to-orange-600 flex items-center justify-center shadow-lg shadow-amber-500/25">
              <span className="text-lg">🧭</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-white text-sm font-bold">"What Should I Do First?" Planner</h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-950 text-amber-300 border border-amber-800">
                  Issue #22
                </span>
              </div>
              <p className="text-slate-400 text-xs">Dynamic sequence optimization based on arrival time & luggage</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/80 flex items-center justify-center transition-colors cursor-pointer"
          >
            ✕
          </button>
        </div>

        {/* control inputs */}
        <div className="px-6 py-3.5 bg-slate-900/60 border-b border-slate-800/60 grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* arrival time */}
          <div>
            <label className="text-[11px] font-mono text-slate-400 block mb-1">Arrival Time:</label>
            <select
              value={arrivalTime}
              onChange={(e) => setArrivalTime(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-amber-500"
            >
              <option value="09:00 AM">09:00 AM (Early Opening)</option>
              <option value="10:30 AM">10:30 AM (Peak Rush)</option>
              <option value="02:00 PM">02:00 PM (Afternoon)</option>
              <option value="04:30 PM">04:30 PM (Late Near Closing)</option>
            </select>
          </div>

          {/* luggage status */}
          <div>
            <label className="text-[11px] font-mono text-slate-400 block mb-1">Luggage Status:</label>
            <div className="flex gap-1.5">
              <button
                type="button"
                onClick={() => setHasHeavyLuggage(true)}
                className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-medium border transition-colors ${
                  hasHeavyLuggage
                    ? 'bg-amber-500/20 text-amber-300 border-amber-500/50'
                    : 'bg-slate-800 text-slate-400 border-slate-700'
                }`}
              >
                🧳 Heavy Suitcases
              </button>
              <button
                type="button"
                onClick={() => setHasHeavyLuggage(false)}
                className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-medium border transition-colors ${
                  !hasHeavyLuggage
                    ? 'bg-amber-500/20 text-amber-300 border-amber-500/50'
                    : 'bg-slate-800 text-slate-400 border-slate-700'
                }`}
              >
                🎒 Backpack Only
              </button>
            </div>
          </div>

          {/* category */}
          <div>
            <label className="text-[11px] font-mono text-slate-400 block mb-1">Category:</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-amber-500"
            >
              <option value="OPEN">OPEN / General</option>
              <option value="OBC-NCL">OBC-NCL</option>
              <option value="GEN-EWS">GEN-EWS</option>
              <option value="SC">SC</option>
              <option value="ST">ST</option>
              <option value="PwD">PwD</option>
            </select>
          </div>
        </div>

        {/* plan content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {loading && (
            <div className="text-center py-10 text-slate-400 text-xs font-mono">
              Calculating optimal queue-aware sequence...
            </div>
          )}

          {planData && !loading && (
            <>
              {/* rationale alert box */}
              <div className="bg-gradient-to-r from-amber-950/40 via-slate-900 to-slate-900 border border-amber-800/40 rounded-xl p-4 shadow-sm">
                <div className="text-xs font-bold text-amber-300 mb-1 flex items-center gap-1.5">
                  <span>📍 Recommended First Stop:</span>
                  <span className="text-white underline">{planData.recommended_first_stop_name}</span>
                </div>
                <div className="space-y-1 text-slate-300 text-xs leading-relaxed mt-2">
                  {planData.rationale.map((r, i) => (
                    <p key={i} className="text-slate-300">
                      {r.replace(/\*\*/g, '')}
                    </p>
                  ))}
                </div>
                <div className="mt-3 flex items-center gap-4 text-[11px] font-mono text-slate-400 pt-2 border-t border-amber-900/40">
                  <span>Walking: <strong className="text-slate-200">{planData.time_breakdown.walking_time}m</strong></span>
                  <span>Queue Wait: <strong className="text-amber-300">{planData.time_breakdown.queue_wait_time}m</strong></span>
                  <span>Service: <strong className="text-slate-200">{planData.time_breakdown.service_time}m</strong></span>
                  <span>Total Est: <strong className="text-emerald-400 font-bold">{planData.estimated_total_time_min} mins</strong></span>
                </div>
              </div>

              {/* step by step list */}
              <div className="space-y-2.5">
                <h4 className="text-xs font-mono text-slate-400 font-bold uppercase tracking-wider">
                  Full Reporting Roadmap ({planData.directions.length} Stations)
                </h4>

                {planData.directions.map((step) => (
                  <div
                    key={step.step_number}
                    className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 flex items-start gap-3 hover:border-slate-700 transition-colors"
                  >
                    <div className="w-6 h-6 rounded-full bg-amber-500/20 text-amber-300 flex items-center justify-center font-mono text-xs font-bold shrink-0 mt-0.5">
                      {step.step_number}
                    </div>

                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <h5 className="text-xs font-bold text-slate-200">
                          {step.title} <span className="text-amber-400/80 font-mono text-[10px]">({step.station_code.toUpperCase()})</span>
                        </h5>
                        <span className="text-[10px] font-mono text-slate-400">
                          ~{step.estimated_wait_min}m queue
                        </span>
                      </div>

                      <p className="text-slate-400 text-xs mt-1 leading-relaxed">
                        {step.action}
                      </p>

                      {step.docs_needed && step.docs_needed.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {step.docs_needed.map((doc, dIdx) => (
                            <span
                              key={dIdx}
                              className="text-[10px] bg-slate-800 text-slate-300 px-2 py-0.5 rounded border border-slate-700/60 font-mono"
                            >
                              📄 {doc}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
