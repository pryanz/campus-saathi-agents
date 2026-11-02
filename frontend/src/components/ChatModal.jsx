import React, { useState, useRef, useEffect } from 'react';

export default function ChatModal({ isOpen, onClose }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: 'Namaste! I am Campus Saathi, your intelligent AI assistant for physical reporting and onboarding.\n\nYou can ask me about:\n• Required documents & cutoff rules (OBC-NCL, EWS, Medical, Gap Certificate)\n• Campus services (Xerox shop, Notary/Affidavit desk, PHC Medical Unit, ATMs)\n• Optimal pathfinding and queue wait times across reporting stations.',
      isStreaming: false
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [isCurrentlyStreaming, setIsCurrentlyStreaming] = useState(false);
  const streamingIntervalRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
    }
  }, [messages, isOpen, loading]);

  useEffect(() => {
    return () => {
      if (streamingIntervalRef.current) {
        clearInterval(streamingIntervalRef.current);
      }
    };
  }, []);

  if (!isOpen) return null;

  // word-by-word streaming effect (dheere dheere aayega text)
  const streamResponse = (fullText, baseMessages) => {
    setIsCurrentlyStreaming(true);
    const words = fullText.split(/(\s+)/);
    let currentIndex = 0;
    let accumulatedText = '';

    setMessages([...baseMessages, { role: 'assistant', text: '', isStreaming: true }]);

    streamingIntervalRef.current = setInterval(() => {
      if (currentIndex < words.length) {
        accumulatedText += words[currentIndex];
        currentIndex++;

        setMessages((prev) => {
          const updated = [...prev];
          if (updated.length > 0 && updated[updated.length - 1].role === 'assistant') {
            updated[updated.length - 1] = {
              role: 'assistant',
              text: accumulatedText,
              isStreaming: true
            };
          }
          return updated;
        });
      } else {
        clearInterval(streamingIntervalRef.current);
        streamingIntervalRef.current = null;
        setIsCurrentlyStreaming(false);

        setMessages((prev) => {
          const updated = [...prev];
          if (updated.length > 0 && updated[updated.length - 1].role === 'assistant') {
            updated[updated.length - 1] = {
              role: 'assistant',
              text: fullText,
              isStreaming: false
            };
          }
          return updated;
        });
      }
    }, 16);
  };

  const handleSkipStreaming = () => {
    if (streamingIntervalRef.current) {
      clearInterval(streamingIntervalRef.current);
      streamingIntervalRef.current = null;
      setIsCurrentlyStreaming(false);
      setMessages((prev) => {
        const updated = [...prev];
        if (updated.length > 0 && updated[updated.length - 1].isStreaming) {
          updated[updated.length - 1].isStreaming = false;
        }
        return updated;
      });
    }
  };

  const handleSend = async (e) => {
    e?.preventDefault();
    const query = input.trim();
    if (!query || loading) return;

    if (isCurrentlyStreaming) {
      handleSkipStreaming();
    }

    const newMessages = [...messages, { role: 'user', text: query, isStreaming: false }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      // pichla chat history context ke liye bhej rahe hain
      const historyPayload = messages.map((m) => ({ role: m.role, text: m.text }));

      const res = await fetch('http://127.0.0.1:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, history: historyPayload })
      });

      const data = await res.json();
      const responseText = data.response || 'No response received from assistant.';
      setLoading(false);
      streamResponse(responseText, newMessages);
    } catch (err) {
      setLoading(false);
      setMessages([
        ...newMessages,
        {
          role: 'assistant',
          text: 'Error connecting to the assistant. Please ensure the backend server is running on port 8000.',
          isStreaming: false
        }
      ]);
    }
  };

  const sampleQueries = [
    'I just arrived with heavy luggage. What should I do first?',
    'What are the cutoff rules for OBC-NCL certificate?',
    'Where can I get photocopies (Xerox) and passport photos?',
    'Where is the Notary on campus for Gap Certificate affidavit?',
    'What if I do not have a JoSAA Medical Certificate?'
  ];

  // formatting aur structured cards ka render
  const formatText = (text, isStreaming) => {
    const lines = text.split('\n');

    return (
      <div className="space-y-1">
        {lines.map((line, idx) => {
          const trimmed = line.trim();

          // section headers (📋, 🏢, 📍, 📅, 🔍, 🧭, ⏱️)
          if (/^(📋|🏢|📍|📅|🔍|🧭|⏱️)\s+\*\*.*?\*\*/.test(trimmed)) {
            const cleanTitle = trimmed.replace(/\*\*/g, '');
            return (
              <div
                key={idx}
                className="bg-gradient-to-r from-cyan-950/70 via-slate-900 to-slate-900/80 border border-cyan-800/50 rounded-xl px-3.5 py-2 my-2 shadow-sm text-cyan-200 font-bold text-xs sm:text-sm flex items-center gap-2"
              >
                <span>{cleanTitle}</span>
              </div>
            );
          }

          // verified source citation card
          if (trimmed.startsWith('[Source:') || trimmed.startsWith('*[Source:')) {
            const cleanCitation = trimmed.replace(/[\*\[\]]/g, '').trim();
            const [srcPart, verPart] = cleanCitation.split(', Verified:');

            return (
              <div
                key={idx}
                className="bg-cyan-950/40 border border-cyan-800/40 rounded-xl p-2.5 my-2.5 flex items-start gap-2 shadow-inner"
              >
                <div className="w-4 h-4 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center shrink-0 mt-0.5">
                  <svg className="w-2.5 h-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <div className="text-[11px] leading-relaxed">
                  <span className="text-slate-400 font-mono">Verified Policy Citation: </span>
                  <span className="text-cyan-300 font-mono font-medium">{srcPart.replace('Source:', '').trim()}</span>
                  {verPart && (
                    <span className="text-slate-400 font-mono block text-[10px] mt-0.5">
                      Last Verified: <span className="text-slate-300">{verPart.trim()}</span>
                    </span>
                  )}
                </div>
              </div>
            );
          }

          const isBullet = trimmed.startsWith('•') || trimmed.startsWith('-');
          const isSubBullet = line.startsWith('  •') || line.startsWith('  -');
          const parts = line.split(/(\*\*.*?\*\*)/g);

          return (
            <div
              key={idx}
              className={`${
                isSubBullet ? 'ml-5 text-slate-300 text-xs sm:text-sm my-0.5' : isBullet ? 'ml-2.5 my-1 text-xs sm:text-sm' : 'my-0.5 text-xs sm:text-sm'
              } leading-relaxed`}
            >
              {parts.map((part, pIdx) => {
                if (part.startsWith('**') && part.endsWith('**')) {
                  return (
                    <strong key={pIdx} className="text-cyan-300 font-semibold">
                      {part.slice(2, -2)}
                    </strong>
                  );
                }
                if (part.includes('➔')) {
                  return (
                    <span key={pIdx} className="text-amber-300 font-medium font-mono">
                      {part}
                    </span>
                  );
                }
                return part;
              })}
            </div>
          );
        })}

        {isStreaming && (
          <span className="inline-block w-2 h-4 bg-cyan-400 animate-pulse ml-1 translate-y-0.5 rounded-sm shadow-sm shadow-cyan-400/50" />
        )}
      </div>
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/75 backdrop-blur-md animate-fadeIn">
      <div className="bg-[#0f172a] border border-slate-800 rounded-2xl w-full max-w-2xl h-[660px] max-h-[92vh] flex flex-col shadow-2xl overflow-hidden relative">
        {/* modal top bar */}
        <div className="px-6 py-4 bg-[#0b1222] border-b border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/25">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-white text-sm font-bold">Campus Saathi Assistant</h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
                  RAG + Streaming
                </span>
              </div>
              <p className="text-slate-400 text-xs">Official Document Verification & Campus Navigation</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {isCurrentlyStreaming && (
              <button
                onClick={handleSkipStreaming}
                className="text-[11px] font-mono bg-slate-800 hover:bg-slate-700 text-cyan-400 px-2.5 py-1 rounded-lg border border-slate-700 transition-colors cursor-pointer"
              >
                Skip
              </button>
            )}

            <button
              onClick={onClose}
              className="w-8 h-8 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/80 flex items-center justify-center transition-colors cursor-pointer"
              aria-label="Close"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* messages container */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[88%] rounded-2xl px-4 py-3 leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-cyan-600 text-white rounded-br-none shadow-md shadow-cyan-900/30 font-medium text-xs sm:text-sm'
                    : 'bg-[#1e293b]/95 text-slate-200 border border-slate-800/80 rounded-bl-none shadow-md'
                }`}
              >
                {msg.role === 'user' ? (
                  <div className="whitespace-pre-wrap">{msg.text}</div>
                ) : (
                  formatText(msg.text, msg.isStreaming)
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-[#1e293b]/90 text-slate-400 border border-slate-800/80 rounded-2xl rounded-bl-none px-4 py-3 text-xs flex items-center gap-2 shadow">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse delay-150" />
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse delay-300" />
                <span className="text-slate-300 font-mono text-[11px] ml-1">
                  Searching institutional knowledge base & simulation tools...
                </span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* ready-made sawal */}
        {messages.length <= 2 && (
          <div className="px-5 py-2.5 border-t border-slate-800/50 bg-[#0b1222]/60">
            <span className="text-[11px] text-slate-400 font-mono block mb-1.5 font-medium">Common Student Queries:</span>
            <div className="flex flex-wrap gap-1.5">
              {sampleQueries.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => setInput(q)}
                  className="text-[11px] bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 hover:text-white px-2.5 py-1 rounded-lg border border-slate-700/60 transition-colors text-left cursor-pointer"
                >
                  "{q}"
                </button>
              ))}
            </div>
          </div>
        )}

        {/* chat input form */}
        <form onSubmit={handleSend} className="p-4 bg-[#0b1222] border-t border-slate-800/80 flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about OBC/EWS rules, Xerox, Medical, Notary, or routing..."
            disabled={loading}
            className="flex-1 bg-slate-900 border border-slate-700/80 rounded-xl px-4 py-2.5 text-xs sm:text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white px-4 py-2.5 rounded-xl text-xs sm:text-sm font-semibold transition-all shadow-md shadow-cyan-900/30 flex items-center gap-1.5 cursor-pointer"
          >
            <span>Send</span>
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
}
