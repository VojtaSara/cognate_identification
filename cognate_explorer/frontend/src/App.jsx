import React, { useState, useRef, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
import TimelinePlugin from "wavesurfer.js/dist/plugins/timeline.esm.js";
import ZoomPlugin from "wavesurfer.js/dist/plugins/zoom.esm.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";
import './App.css';

export default function App() {
  const [audioUrl, setAudioUrl] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [segments, setSegments] = useState([]);
  const [transcript, setTranscript] = useState(null);
  const [similarityAnalysis, setSimilarityAnalysis] = useState(null);
  const [referenceFile, setReferenceFile] = useState(null);
  const [longAudioFile, setLongAudioFile] = useState(null);
  const wavesurferRef = useRef(null);
  const waveformContainerRef = useRef(null);
  const pollingRef = useRef(null);

  const handleReferenceFile = (files) => {
    const file = files[0];
    if (!file) return;
    setReferenceFile(file);
  };

  const handleLongAudioFile = (files) => {
    const file = files[0];
    if (!file) return;
    setLongAudioFile(file);
    
    // Set the long audio for waveform display
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    setSegments([]);
  };

  const handleUpload = () => {
    if (!referenceFile || !longAudioFile) {
      alert("Please select both reference and long audio files");
      return;
    }

    setStatus("uploading");

    const formData = new FormData();
    formData.append("reference_audio", referenceFile);
    formData.append("long_audio", longAudioFile);

    fetch("http://localhost:8000/api/upload", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        setJobId(data.job_id);
        setStatus("processing");
        startPolling(data.job_id);
      })
      .catch((err) => {
        console.error("Upload error:", err);
        setStatus("error");
      });
  };

  const startPolling = (jobId) => {
    pollingRef.current = setInterval(() => {
      fetch(`http://localhost:8000/api/status/${jobId}`)
        .then((res) => res.json())
        .then((data) => {
          if (data.status === "done") {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
            setStatus("done");
            setSegments(data.result.whisper.words);
            setTranscript(data.result.whisper);
            setSimilarityAnalysis(data.result.similarity_analysis);
          } else if (data.status === "error") {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
            setStatus("error");
          }
        })
        .catch((err) => {
          console.error("Polling error:", err);
          setStatus("error");
        });
    }, 1000);
  };

  const cancelPolling = () => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
      setStatus("cancelled");
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const files = e.dataTransfer.files;
    handleLongAudioFile(files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  useEffect(() => {
    if (audioUrl && waveformContainerRef.current) {
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
      }

      wavesurferRef.current = WaveSurfer.create({
        container: waveformContainerRef.current,
        waveColor: "#e2e8f0",
        progressColor: "#667eea",
        height: 100,
        responsive: true,
        url: audioUrl,
        plugins: [
          TimelinePlugin.create({ container: "#wave-timeline" }),
          ZoomPlugin.create({ minZoom: 10, maxZoom: 1000 }),
          RegionsPlugin.create(),
        ],
      });
    }
  }, [audioUrl]);

  useEffect(() => {
    if (wavesurferRef.current && segments.length > 0) {
      const regionsPlugin = wavesurferRef.current.getActivePlugins().regions;
      if (regionsPlugin) {
        regionsPlugin.clearRegions();

        segments.forEach((word) => {
          const region = regionsPlugin.addRegion({
            start: word.start,
            end: word.end,
            color: "rgba(102, 126, 234, 0.2)",
            data: { text: word.text },
          });

          if (region.element) {
            const tooltip = document.createElement("div");
            tooltip.textContent = word.text;
            tooltip.style.position = "absolute";
            tooltip.style.fontSize = "10px";
            tooltip.style.padding = "4px 8px";
            tooltip.style.borderRadius = "6px";
            tooltip.style.background = "rgba(0,0,0,0.8)";
            tooltip.style.color = "white";
            tooltip.style.pointerEvents = "none";
            tooltip.style.transform = "translateY(-120%)";
            tooltip.style.fontFamily = "'Inter', sans-serif";
            region.element.appendChild(tooltip);

            region.element.addEventListener("mouseenter", () => {
              tooltip.style.display = "block";
            });
            region.element.addEventListener("mouseleave", () => {
              tooltip.style.display = "none";
            });
          }
        });
      }
    }
  }, [segments]);

  return (
    <div className="app-container">
      {/* Header */}
      <header>
        <h1>In-Audio Cognate Analyzer</h1>
        <div>Powered by AI</div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Upload Section */}
        <section className="section">
          <h2>1 <span style={{fontWeight:400}}>Upload Audio Files</span></h2>
          {/* Reference Audio Upload */}
          <label>Reference Audio <span style={{fontWeight:400, fontSize:'0.95em'}}>(short, ~500ms)</span></label>
          <input type="file" accept="audio/*" onChange={(e) => handleReferenceFile(e.target.files)} id="reference-upload" style={{display:'none'}} />
          <label htmlFor="reference-upload" className="upload-label">
            <svg height="24" width="24" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
            <span>{referenceFile ? referenceFile.name : "Choose reference audio"}</span>
          </label>

          {/* Long Audio Upload */}
          <label>Long Audio <span style={{fontWeight:400, fontSize:'0.95em'}}>(to analyze)</span></label>
          <input type="file" accept="audio/*" onChange={(e) => handleLongAudioFile(e.target.files)} id="long-upload" style={{display:'none'}} />
          <label htmlFor="long-upload" className="upload-label">
            <svg height="24" width="24" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
            <span>{longAudioFile ? longAudioFile.name : "Choose long audio"}</span>
          </label>

          {/* Upload Button */}
          <button onClick={handleUpload} disabled={!referenceFile || !longAudioFile}>
            <svg height="20" width="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{verticalAlign:'middle',marginRight:6}}><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
            Analyze Similarity
          </button>

          {/* Status Indicators */}
          {(status === "uploading" || status === "processing") && (
            <div className="status">
              <span>{status === "uploading" ? "Uploading files..." : "Processing audio..."}</span>
              <br/>
              <span style={{fontSize:'0.95em',color:'#666'}}>This may take a few moments</span>
              {status === "processing" && (
                <div style={{marginTop:8}}>
                  <button onClick={cancelPolling} style={{background:'#eee',color:'#991b1b'}}>Cancel</button>
                </div>
              )}
            </div>
          )}
          {status === "error" && (
            <div className="status error">
              <svg height="20" width="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{verticalAlign:'middle',marginRight:6}}><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
              Processing failed
            </div>
          )}
        </section>

        {/* Audio Player Section */}
        <section className="section">
          <h2>2 <span style={{fontWeight:400}}>Audio Player</span></h2>
          <div ref={waveformContainerRef} style={{marginBottom:12}} />
          <div id="wave-timeline" style={{marginBottom:18}} />
          {/* Player Controls */}
          <div style={{display:'flex',gap:12,marginBottom:18}}>
            <button onClick={() => wavesurferRef.current && wavesurferRef.current.play()} title="Play" style={{background:'#10b981'}}>
              <svg height="24" width="24" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
            </button>
            <button onClick={() => wavesurferRef.current && wavesurferRef.current.pause()} title="Pause" style={{background:'#f59e42'}}>
              <svg height="24" width="24" fill="currentColor" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
            </button>
            <button onClick={() => {if (wavesurferRef.current) {wavesurferRef.current.pause();wavesurferRef.current.seekTo(0);}}} title="Stop" style={{background:'#ef4444'}}>
              <svg height="24" width="24" fill="currentColor" viewBox="0 0 24 24"><path d="M6 6h12v12H6z"/></svg>
            </button>
          </div>
          {/* Zoom Controls */}
          <div style={{display:'flex',gap:8}}>
            <button onClick={() => wavesurferRef.current?.zoom(100)} style={{background:'#3b82f6'}}>Zoom In</button>
            <button onClick={() => wavesurferRef.current?.zoom(10)} style={{background:'#888'}}>Zoom Out</button>
          </div>
        </section>
      </main>

      {/* Results Section */}
      {status === "done" && transcript && (
        <section className="section">
          <h2>3 <span style={{fontWeight:400}}>Similarity Analysis Results</span></h2>
          {similarityAnalysis ? (
            <div>
              <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:8}}>
                <svg height="20" width="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                <span>Reference Audio: User-provided reference (500ms)</span>
              </div>
              <div className="similarity-legend">
                <span className="similarity-hot">Very Hot</span>
                <span className="similarity-warm">Hot</span>
                <span className="similarity-mid">Warm</span>
                <span className="similarity-cool">Cool</span>
                <span className="similarity-cold">Cold</span>
              </div>
            </div>
          ) : (
            <div className="status">
              <svg height="20" width="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" /></svg>
              No similarity analysis data received
            </div>
          )}

          <div className="transcript">
            <h3 style={{fontWeight:600,marginBottom:10}}>Transcript with Similarity Analysis</h3>
            <div>
              {(() => {
                // Debug: Log what we're receiving
                console.log("Similarity analysis data:", similarityAnalysis);
                
                // Calculate normalized distances for color mapping
                let maxDistance = 0;
                let minDistance = Infinity;
                const wordDistances = {};
                
                if (similarityAnalysis && similarityAnalysis.similarities) {
                  // Find min/max distances for normalization
                  similarityAnalysis.similarities.forEach(item => {
                    if (item.distance !== null && item.distance !== undefined) {
                      maxDistance = Math.max(maxDistance, item.distance);
                      minDistance = Math.min(minDistance, item.distance);
                    }
                  });
                  
                  // Create distance mapping for each word
                  transcript.words.forEach((word, idx) => {
                    const similarityInfo = similarityAnalysis.similarities.find(
                      s => s.word === word.word && s.start === word.start
                    );
                    if (similarityInfo && similarityInfo.distance !== null) {
                      wordDistances[idx] = similarityInfo.distance;
                    } else {
                      wordDistances[idx] = maxDistance; // Default to max distance
                    }
                  });
                }
                
                // Function to get color based on normalized distance
                const getDistanceClass = (distance) => {
                  // If no similarity data, use default colors
                  if (!similarityAnalysis || !similarityAnalysis.similarities) {
                    return ''; // Gray
                  }
                  
                  if (maxDistance === minDistance) return ''; // Gray
                  
                  // Normalize distance to 0-1 range (0 = hot/close, 1 = cold/far)
                  const normalized = (distance - minDistance) / (maxDistance - minDistance);
                  
                  // Color gradient from hot (red) to cold (blue)
                  if (normalized < 0.2) return 'similarity-hot';      // Red (Very hot)
                  if (normalized < 0.4) return 'similarity-warm';      // Orange (Hot)
                  if (normalized < 0.6) return 'similarity-mid';      // Yellow (Warm)
                  if (normalized < 0.8) return 'similarity-cool';      // Blue (Cool)
                  return 'similarity-cold';                            // Indigo (Cold)
                };
                
                return (
                  <div>
                    {transcript.words.map((word, idx) => {
                      const distance = wordDistances[idx] || 0;
                      const className = getDistanceClass(distance);
                      
                      return (
                        <span
                          key={idx}
                          className={className}
                          title={`Distance: ${distance.toFixed(3)}`}
                          onClick={() => {
                            if (wavesurferRef.current) {
                              wavesurferRef.current.seekTo(word.start / wavesurferRef.current.getDuration());
                              wavesurferRef.current.play();
                            }
                          }}
                        >
                          {word.word}
                        </span>
                      );
                    })}
                  </div>
                );
              })()}
            </div>
            
            <div style={{marginTop:12,fontSize:'0.98em',color:'#666'}}>
              <div>Language: {transcript.metadata?.language}</div>
              <div>Language probability: {transcript.metadata?.language_probability?.toFixed(2)}</div>
              <div>Timestamp: {transcript.metadata?.timestamp}</div>
            </div>
            
            {similarityAnalysis && similarityAnalysis.similarities && (
              <div style={{marginTop:16}}>
                <h4 style={{fontWeight:600,marginBottom:6}}>Top Similar Words</h4>
                <div>
                  {similarityAnalysis.similarities.slice(0, 10).map((item, idx) => (
                    <div key={idx} style={{display:'flex',justifyContent:'space-between',alignItems:'center',fontSize:'0.98em',marginBottom:2}}>
                      <span>{item.word}</span>
                      <span style={{
                        backgroundColor: item.distance < 0.5 ? '#dcfce7' : 
                                       item.distance < 1.0 ? '#fef3c7' : '#fee2e2',
                        color: item.distance < 0.5 ? '#166534' : 
                               item.distance < 1.0 ? '#92400e' : '#991b1b',
                        borderRadius: 5,
                        padding: '2px 8px',
                        marginLeft: 8
                      }}>
                        d={item.distance?.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
}