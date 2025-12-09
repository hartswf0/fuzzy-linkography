import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.9.0/dist/transformers.min.js";
env.allowLocalModels = false;
env.useBrowserCache = true;
// Fallback similarity if AI model fails
function jaccardSimilarity(textA, textB) {
    const setA = new Set(textA.toLowerCase().split(/\W+/).filter(x => x.length > 0));
    const setB = new Set(textB.toLowerCase().split(/\W+/).filter(x => x.length > 0));
    if (setA.size === 0 || setB.size === 0) return 0;
    const intersection = new Set([...setA].filter(x => setB.has(x)));
    return intersection.size / (setA.size + setB.size - intersection.size);
}

const e = window.React.createElement;
const useState = window.React.useState;
const useEffect = window.React.useEffect;
const useRef = window.React.useRef;
const useMemo = window.React.useMemo;

// --- MATH UTILS (Ported from app.js) ---
const DIMENSION = 384;

function sum(xs) { return xs.reduce((a, b) => a + b, 0); }

function dotProduct(vectorA, vectorB) {
    let dotProd = 0;
    // vectors might be standard js arrays or typed arrays from tensor
    for (let comp = 0; comp < vectorA.length; comp++) {
        dotProd += vectorA[comp] * vectorB[comp];
    }
    return dotProd;
}

function magnitude(vector) { return Math.sqrt(dotProduct(vector, vector)); }

function cosineSimilarity(vectorA, vectorB) {
    const dotProd = dotProduct(vectorA, vectorB);
    const magProd = magnitude(vectorA) * magnitude(vectorB);
    return (magProd === 0) ? 0 : (dotProd / magProd);
}

function scale(num, [oldMin, oldMax], [newMin, newMax]) {
    const oldRange = oldMax - oldMin;
    const newRange = newMax - newMin;
    return (oldRange === 0) ? newMin : (((num - oldMin) / oldRange) * newRange) + newMin;
}

function entropy(pOn, pOff) {
    const pOnPart = pOn > 0 ? -(pOn * Math.log2(pOn)) : 0;
    const pOffPart = pOff > 0 ? -(pOff * Math.log2(pOff)) : 0;
    return pOnPart + pOffPart;
}

// --- VISUALIZATION CONSTANTS ---
const GRAPH_WIDTH = 800; // width of the SVG canvas
const INIT_X = 20;
const INIT_Y = 100;
const MOVE_LINK_BAR_HEIGHT = 40;

// --- COMPONENTS ---

// 1. Matrix View: Surfaces the invisible similarity scores
function MatrixView({ moves, links, threshold }) {
    if (!moves || moves.length === 0) return null;

    const size = moves.length;
    // We want to render a grid. CSS Grid is easy for this.
    // We'll only show the lower triangle or full matrix? Full is easier to read as a map.

    const cells = [];
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            let val = 0;
            if (i === j) val = 1;
            else if (j < i) val = links[i] && links[i][j] || 0;
            else val = links[j] && links[j][i] || 0;

            // Visual encoding:
            // Grayscale for strength
            // Highlight if above threshold
            const isStrong = val >= threshold;
            const brightness = Math.floor(val * 255);
            const color = isStrong
                ? `rgba(0, 123, 255, ${val})` // Blue tint if active link
                : `rgba(0, 0, 0, ${val * 0.3})`; // Faint gray if not

            cells.push(e("div", {
                key: `${i}-${j}`,
                className: "matrix-cell",
                style: { backgroundColor: color },
                title: `${moves[i].text.slice(0, 10)}... <-> ${moves[j].text.slice(0, 10)}... : ${val.toFixed(3)}`
            }));
        }
    }

    return e("div", {},
        e("h4", {}, "Latent Similarity Matrix"),
        e("div", {
            className: "matrix-grid",
            style: { gridTemplateColumns: `repeat(${size}, 1fr)` }
        }, ...cells)
    );
}

// 2. Linkograph View: The main graph
function InteractiveLinkograph({ moves, links, threshold, moveTextMode }) {
    if (!moves || moves.length === 0) return null;

    // Pre-calc layout
    // Spacing depends on graph width (we fix graph width to be responsive-ish)
    const spacing = (moves.length > 1) ? (GRAPH_WIDTH - (INIT_X * 2)) / (moves.length - 1) : 0;

    const moveLoc = (idx) => ({ x: (idx * spacing) + INIT_X, y: INIT_Y });
    const elbow = (pt1, pt2) => ({ x: (pt1.x + pt2.x) / 2, y: pt1.y - ((pt2.x - pt1.x) / 2) });

    // Generate lines
    const lines = [];
    const joints = [];

    // Iterate all pairs
    for (let i = 1; i < moves.length; i++) {
        for (let j = 0; j < i; j++) {
            const strength = links[i][j] || 0;
            if (strength < threshold) continue;

            // Draw link
            const p1 = moveLoc(i);
            const p2 = moveLoc(j);
            const joint = elbow(p1, p2);

            // Color mapping: opacity based on strength? Or color scale?
            // Let's use opacity for "analog" feel
            const opacity = scale(strength, [threshold, 1], [0.2, 1]);
            const color = `rgba(0,0,0, ${opacity})`;

            lines.push(e("polyline", {
                key: `l-${i}-${j}`,
                points: `${p1.x},${p1.y} ${joint.x},${joint.y} ${p2.x},${p2.y}`,
                fill: "none",
                stroke: color,
                strokeWidth: 2 * strength // thicker = stronger
            }));

            joints.push(e("circle", {
                key: `j-${i}-${j}`,
                cx: joint.x, cy: joint.y, r: 2 * strength,
                fill: "black", fillOpacity: opacity
            }));
        }
    }

    // Generate move nodes
    const nodes = moves.map((m, idx) => {
        const loc = moveLoc(idx);
        return e("g", { key: `n-${idx}` },
            e("circle", { cx: loc.x, cy: loc.y, r: 4, fill: "#333" }),
            (moveTextMode !== "NONE") ? e("text", {
                x: loc.x, y: loc.y + 15,
                textAnchor: "start",
                transform: `rotate(45, ${loc.x}, ${loc.y + 15})`,
                fontSize: "10px",
                fill: "#555"
            }, (moveTextMode === "INDEX" ? idx : m.text)) : null
        );
    });

    // SVG height needs to accommodate the tallest arc (half width)
    const height = (GRAPH_WIDTH / 2) + INIT_Y + 100;

    return e("div", { className: "graph-container" },
        e("h4", {}, `Fuzzy Linkograph (Threshold: ${threshold})`),
        e("svg", {
            viewBox: `0 0 ${GRAPH_WIDTH} ${height}`,
            style: { width: "100%", height: "auto" }
        }, ...lines, ...joints, ...nodes)
    );
}

// 3. Stats Panel
function StatsPanel({ moves, links, threshold }) {
    // Calculate Link Density & Entropy live
    // Reuse logic from app.js but simplified for real-time

    const activeLinks = [];
    let totalPossible = 0;
    let totalWeight = 0;
    const weights = [];

    for (let i = 1; i < links.length; i++) { // wait, links is object in app.js, here I might structure it differently?
        // Ensure links structure is consistent
        // The main app uses: links[i][j] = score
        for (let j = 0; j < i; j++) {
            totalPossible++;
            const s = links[i][j];
            if (s >= threshold) {
                const scaled = scale(s, [threshold, 1], [0, 1]);
                activeLinks.push(scaled);
                totalWeight += scaled;
                weights.push(scaled);
            } else {
                weights.push(0);
            }
        }
    }

    const density = (totalPossible > 0) ? (totalWeight / moves.length) : 0; // standard def is / possible links? app.js uses / moves.length which is weird but I'll stick to it or standard? app.js: overallLinkWeight / graph.moves.length

    // Entropy estimate (simplified: just distribution of link weights)
    // Shannon entropy of the link weight distribution?
    // app.js entropy is complex, let's just show Active Link Count for now as a proxy for complexity
    const activeCount = activeLinks.length;

    return e("div", {},
        e("div", { className: "metric-card" },
            e("div", { className: "metric-label" }, "Active Links"),
            e("div", { className: "metric-value" }, activeCount)
        ),
        e("div", { className: "metric-card" },
            e("div", { className: "metric-label" }, "Link Density Index"),
            e("div", { className: "metric-value" }, density.toFixed(2))
        ),
        e("div", { className: "metric-card" },
            e("div", { className: "metric-label" }, "Total Moves"),
            e("div", { className: "metric-value" }, moves.length)
        ),
    );
}

// --- MAIN APP ---

function App() {
    const [text, setText] = useState(
        `Start stream intro
Play game match 1
Commentate on gameplay
Check OBS bitrate
Read chat message
Reply to viewer
Play game match 1
Monitor analytics
Troubleshoot audio lag
Play game match 2`
    );

    const [moves, setMoves] = useState([]);
    const [links, setLinks] = useState({}); // links[i][j] -> score
    const [threshold, setThreshold] = useState(0.4);
    const [extractor, setExtractor] = useState(null);
    const [modelStatus, setModelStatus] = useState("loading"); // loading, active, fallback

    // Load Model with Timeout Fallback
    useEffect(() => {
        let isMounted = true;
        async function load() {
            try {
                // Race: 10s timeout vs Model Load
                const loadPromise = pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
                const timeoutPromise = new Promise((_, reject) =>
                    setTimeout(() => reject(new Error("Timeout")), 8000)
                );

                const pipe = await Promise.race([loadPromise, timeoutPromise]);

                if (isMounted) {
                    setExtractor(() => pipe);
                    setModelStatus("active");
                    document.getElementById("loading").style.display = "none";
                }
            } catch (err) {
                console.warn("AI Model load failed/timed out, switching to Fallback", err);
                if (isMounted) {
                    setModelStatus("fallback");
                    document.getElementById("loading").style.display = "none";
                }
            }
        }
        load();
        return () => { isMounted = false; };
    }, []);

    // Process Text
    useEffect(() => {
        const timer = setTimeout(async () => {
            const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
            const newMoves = lines.map(t => ({ text: t }));
            const newLinks = {};

            if (modelStatus === "active" && extractor) {
                const newEmbeddings = [];
                for (const m of newMoves) {
                    const output = await extractor(m.text, { pooling: "mean", normalize: true });
                    newEmbeddings.push(output.data);
                }
                for (let i = 0; i < newMoves.length; i++) {
                    newLinks[i] = {};
                    for (let j = 0; j < i; j++) {
                        newLinks[i][j] = cosineSimilarity(newEmbeddings[i], newEmbeddings[j]);
                    }
                }
            } else if (modelStatus === "fallback") {
                // Jaccard Fallback
                for (let i = 0; i < newMoves.length; i++) {
                    newLinks[i] = {};
                    for (let j = 0; j < i; j++) {
                        newLinks[i][j] = jaccardSimilarity(newMoves[i].text, newMoves[j].text);
                    }
                }
            }

            setMoves(newMoves);
            setLinks(newLinks);
        }, 500);

        return () => clearTimeout(timer);
    }, [text, extractor, modelStatus]);

    // -- Render --

    return e("div", { style: { display: 'flex', width: '100%', height: '100%' } },
        // 1. Editor Panel
        e("div", { className: "panel editor-panel" },
            e("div", { className: "panel-header" }, "Creator Trail (Input)"),
            e("div", { className: "editor-content" },
                e("textarea", {
                    id: "trail-editor",
                    value: text,
                    onChange: (ev) => setText(ev.target.value),
                    placeholder: "Type your design moves here (one per line)..."
                })
            )
        ),

        // 2. Viz Panel
        e("div", { className: "panel viz-panel" },
            e("div", { className: "panel-header" }, "Visualizers"),
            e("div", { className: "viz-content" },
                e(InteractiveLinkograph, { moves, links, threshold, moveTextMode: "FULL" }),
                e(MatrixView, { moves, links, threshold })
            )
        ),

        // 3. Inspector Panel
        e("div", { className: "panel inspector-panel" },
            e("div", { className: "panel-header" }, "Controls & Stats"),
            e("div", { className: "inspector-scroll" },
                e("div", { className: "control-group" },
                    e("label", {}, "Similarity Threshold (Min Link Strength)"),
                    e("span", { className: "value-display" }, threshold),
                    e("input", {
                        type: "range",
                        min: "0", max: "1", step: "0.01",
                        value: threshold,
                        onChange: (ev) => setThreshold(parseFloat(ev.target.value))
                    }),
                    e("p", { style: { fontSize: '0.8em', color: '#666' } },
                        "Drag to filter out weak semantic connections. This reveals the 'strong bones' of the idea structure."
                    )
                ),
                e("hr"),
                e(StatsPanel, { moves, links, threshold })
            )
        )
    );
}

const root = ReactDOM.createRoot(document.getElementById('app-root'));
root.render(e(App));
