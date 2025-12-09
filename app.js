// import embeddings lib
import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.9.0/dist/transformers.min.js";
const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
const DIMENSION = 384;
// import react
const e = React.createElement;

/// global graph properties

const MOVE_TEXT_MODE = "NONE"; // or "FULL" or "INDEX"
const SHOULD_COLORIZE_LINKS = true;
const GRAPH_WIDTH = 1000;
const INIT_X = 10;
const INIT_Y = { FULL: 500, INDEX: 80, NONE: 60 }[MOVE_TEXT_MODE];
const MOVE_LINK_BAR_HEIGHT = 40; // how tall the forelink/backlink bars over each move should be
const MIN_LINK_STRENGTH = 0.35;
const SEGMENT_THRESHOLD = 1000 * 60 * 30; // 30 mins -> milliseconds

/// math utils

function sum(xs) {
	return xs.reduce((a, b) => a + b, 0);
}

function dotProduct(vectorA, vectorB) {
	let dotProd = 0;
	for (let comp = 0; comp < DIMENSION; comp++) {
		dotProd += vectorA[comp] * vectorB[comp];
	}
	return dotProd;
}

function magnitude(vector) {
	return Math.sqrt(dotProduct(vector, vector));
}

function cosineSimilarity(vectorA, vectorB) {
	const dotProd = dotProduct(vectorA, vectorB);
	const magProd = magnitude(vectorA) * magnitude(vectorB);
	return (magProd === 0) ? 0 : (dotProd / magProd);
}

// Translate `num` from the scale bounded by `oldMin` and `oldMax`
// to the scale bounded by `newMin` and `newMax`.
function scale(num, [oldMin, oldMax], [newMin, newMax]) {
	const oldRange = oldMax - oldMin;
	const newRange = newMax - newMin;
	return (oldRange === 0) ? newMin : (((num - oldMin) / oldRange) * newRange) + newMin;
}

/// stats on a computed linkograph

// Given `linkStrengths`, a list of raw (unscaled) semantic similarity scores:
// 1. Threshold each semantic similarity value on `MIN_LINK_STRENGTH`.
// 2. Scale it from the range `[0, 1]` to the range `[MIN_LINK_STRENGTH, 1]`.
// 3. Sum up all the scaled values and return the sum.
function totalLinkWeight(linkStrengths) {
	return sum(
		linkStrengths
			.filter(n => n >= MIN_LINK_STRENGTH)
			.map(n => scale(n, [MIN_LINK_STRENGTH, 1], [0, 1]))
	);
}

function computeLinkDensityIndex(graph) {
	const overallLinkWeight = sum(Object.values(graph.links).map(
		linkSet => totalLinkWeight(Object.values(linkSet)))
	);
	graph.linkDensityIndex = overallLinkWeight / graph.moves.length;
}

function computeMoveWeights(graph) {
	for (let i = 0; i < graph.moves.length; i++) {
		const backlinkStrengths = Object.values(graph.links[i]);
		graph.moves[i].backlinkWeight = totalLinkWeight(backlinkStrengths) || 0;
		const forelinkStrengths = Object.values(graph.links).map(linkSet => linkSet[i] || 0);
		graph.moves[i].forelinkWeight = totalLinkWeight(forelinkStrengths) || 0;
	}
	// naïvely mark critical moves: top 3 link weights in each direction
	graph.moves.toSorted((a, b) => b.backlinkWeight - a.backlinkWeight).slice(0, 3)
		.forEach(move => { move.backlinkCriticalMove = true; });
	graph.moves.toSorted((a, b) => b.forelinkWeight - a.forelinkWeight).slice(0, 3)
		.forEach(move => { move.forelinkCriticalMove = true; });
	// calculate max forelink and backlink weights seen, for visual scaling
	graph.maxForelinkWeight = Math.max(...graph.moves.map(move => move.forelinkWeight)) || 0;
	graph.maxBacklinkWeight = Math.max(...graph.moves.map(move => move.backlinkWeight)) || 0;
}

function entropy(pOn, pOff) {
	const pOnPart = pOn > 0 ? -(pOn * Math.log2(pOn)) : 0;
	const pOffPart = pOff > 0 ? -(pOff * Math.log2(pOff)) : 0;
	return pOnPart + pOffPart;
}

function computeEntropy(graph) {
	// backlinks and forelinks
	for (let i = 0; i < graph.moves.length; i++) {
		// backlinks
		const maxPossibleBacklinkWeight = i;
		const backlinkPOn = graph.moves[i].backlinkWeight / maxPossibleBacklinkWeight;
		const backlinkPOff = 1 - backlinkPOn;
		graph.moves[i].backlinkEntropy = entropy(backlinkPOn, backlinkPOff);
		// forelinks
		const maxPossibleForelinkWeight = graph.moves.length - (i + 1);
		const forelinkPOn = graph.moves[i].forelinkWeight / maxPossibleForelinkWeight;
		const forelinkPOff = 1 - forelinkPOn;
		graph.moves[i].forelinkEntropy = entropy(forelinkPOn, forelinkPOff);
	}
	graph.backlinkEntropy = sum(graph.moves.map(move => move.backlinkEntropy));
	graph.forelinkEntropy = sum(graph.moves.map(move => move.forelinkEntropy));
	// horizonlinks
	// Each "horizon state" is the set of possible links between pairs of moves
	// that are N apart from each other. In traditional linkography, the one-link
	// horizon state at the max possible value of N is skipped, because it always
	// has entropy of exactly 0 (the singular link either exists or doesn't, and
	// is fully "predictable" either way). For fuzzy linkography, however, we
	// include this state in the calculation, because that link may be of some
	// unknown nonzero strength and therefore some unknown nonzero entropy value.
	graph.horizonlinkEntropy = 0;
	for (let horizon = 1; horizon < graph.moves.length; horizon++) {
		// get all pairs of move indexes (i,j) that are N apart
		const moveIndexPairs = [];
		for (let i = 0; i < graph.moves.length - horizon; i++) {
			moveIndexPairs.push([i, i + horizon]);
		}
		// get total link weight between these pairs
		const horizonlinkStrengths = moveIndexPairs.map(([i, j]) => graph.links[j]?.[i] || 0);
		const horizonlinkWeight = totalLinkWeight(horizonlinkStrengths);
		// calculate entropy
		const horizonlinkPOn = horizonlinkWeight / moveIndexPairs.length;
		const horizonlinkPOff = 1 - horizonlinkPOn;
		graph.horizonlinkEntropy += entropy(horizonlinkPOn, horizonlinkPOff);
	}
	// sum them all up
	graph.entropy = graph.backlinkEntropy + graph.forelinkEntropy + graph.horizonlinkEntropy;
}

function computeActorLinkStats(graph) {
	const linkStrengthsByActorPair = {};
	const possibleLinkCountsByActorPair = {};
	graph.copyCount = 0;
	for (let i = 0; i < graph.moves.length - 1; i++) {
		const actorA = graph.moves[i]?.actor || 0;
		for (let j = i + 1; j < graph.moves.length; j++) {
			// skip ~verbatim copies
			if ((graph.links[j]?.[i] || 0) >= 0.99) {
				graph.copyCount++;
				continue;
			}
			const actorB = graph.moves[j]?.actor || 0;
			// track the raw link weights of backlinks from actor B to actor A
			const pair = [actorB, actorA].join(":");
			if (!linkStrengthsByActorPair[pair]) {
				linkStrengthsByActorPair[pair] = [];
			}
			linkStrengthsByActorPair[pair].push(graph.links[j]?.[i] || 0);
			// increment count of possible backlinks from actor B to actor A
			if (!possibleLinkCountsByActorPair[pair]) {
				possibleLinkCountsByActorPair[pair] = 0;
			}
			possibleLinkCountsByActorPair[pair] += 1;
		}
	}
	const linkDensitiesByActorPair = {};
	for (const [pair, strengths] of Object.entries(linkStrengthsByActorPair)) {
		linkDensitiesByActorPair[pair] = totalLinkWeight(strengths) / possibleLinkCountsByActorPair[pair];
	}
	graph.linkDensitiesByActorPair = linkDensitiesByActorPair;
}

/// data processing

async function embed(str) {
	return (await extractor(str, { convert_to_tensor: "True", pooling: "mean", normalize: true }))[0];
}

async function computeLinks(moves) {
	// calculate embeddings for each move
	for (const move of moves) {
		move.embedding = (await embed(move.text)).tolist();
	}
	// build links table
	const links = {};
	for (let i = 0; i < moves.length; i++) {
		const currMove = moves[i];
		links[i] = {};
		for (let j = 0; j < i; j++) {
			const prevMove = moves[j];
			links[i][j] = cosineSimilarity(currMove.embedding, prevMove.embedding);
		}
	}
	return links;
}

/// react rendering

// Given the rendered locations of two design moves, return the location of the
// right-angle "elbow" between them.
function elbow(pt1, pt2) {
	const x = (pt1.x + pt2.x) / 2;
	const y = pt1.y - ((pt2.x - pt1.x) / 2);
	return { x, y };
}

// Given `props` augmented with the `idx` of a design move, return the location
// at which this move should be rendered.
function moveLoc(props) {
	return { x: (props.idx * props.moveSpacing) + INIT_X, y: INIT_Y };
}

function shouldSegmentTimeline(currMove, prevMove) {
	if (!(currMove?.timestamp && prevMove?.timestamp)) return false;
	const deltaTime = Date.parse(currMove.timestamp) - Date.parse(prevMove.timestamp);
	return deltaTime >= SEGMENT_THRESHOLD;
}

function DesignMove(props) {
	const move = props.moves[props.idx];
	const currLoc = moveLoc(props);
	const scaledForelinkWeight = scale(
		move.forelinkWeight, [0, props.maxForelinkWeight], [0, MOVE_LINK_BAR_HEIGHT]
	);
	const scaledBacklinkWeight = scale(
		move.backlinkWeight, [0, props.maxBacklinkWeight], [0, MOVE_LINK_BAR_HEIGHT]
	);
	const moveLinkBarSize = 10 + MOVE_LINK_BAR_HEIGHT + 10;
	// create correct move marker based on actor
	let moveMarker = null;
	if (move.actor === 1) {
		moveMarker = e("rect", {
			x: currLoc.x - 5, y: currLoc.y - 5, height: 10, width: 10, fill: "blue"
		});
	}
	else {
		moveMarker = e("circle", { cx: currLoc.x, cy: currLoc.y, r: 5, fill: "red" });
	}
	return e("g", {},
		((MOVE_TEXT_MODE === "FULL") ? e("text", {
			x: currLoc.x + 5, y: currLoc.y - moveLinkBarSize,
			transform: `rotate(270, ${currLoc.x + 5}, ${currLoc.y - moveLinkBarSize})`,
			fontWeight: (move.backlinkCriticalMove || move.forelinkCriticalMove) ? "bold" : "normal",
		}, move.text) : null),
		((MOVE_TEXT_MODE === "INDEX") ? e("text", {
			x: currLoc.x, y: currLoc.y - moveLinkBarSize,
			textAnchor: "middle",
			fontWeight: (move.backlinkCriticalMove || move.forelinkCriticalMove) ? "bold" : "normal",
		}, props.idx) : null),
		e("rect", {
			x: currLoc.x - 5, y: (currLoc.y - 10) - scaledBacklinkWeight,
			width: 5, height: scaledBacklinkWeight, fill: "#998ec3",
		}),
		e("rect", {
			x: currLoc.x, y: (currLoc.y - 10) - scaledForelinkWeight,
			width: 5, height: scaledForelinkWeight, fill: "#f1a340",
		}),
		moveMarker
	);
}

function makeTimelineDividers(props) {
	const dividers = [];
	for (let idx = 0; idx < props.moves.length; idx++) {
		const currLoc = moveLoc({ ...props, idx });
		const splitAfter = shouldSegmentTimeline(props.moves[idx + 1], props.moves[idx]);
		if (!splitAfter) continue;
		dividers.push(e("line", {
			stroke: "#999", strokeDasharray: "2", strokeWidth: 1,
			x1: currLoc.x + (props.moveSpacing / 2), y1: currLoc.y - GRAPH_WIDTH,
			x2: currLoc.x + (props.moveSpacing / 2), y2: currLoc.y + GRAPH_WIDTH,
		}));
	}
	return dividers;
}

function makeLinkObjects(props) {
	const linkLines = [];
	const linkJoints = [];
	for (const [currIdx, linkSet] of Object.entries(props.links)) {
		const currLoc = moveLoc({ ...props, idx: currIdx });
		for (const [prevIdx, strength] of Object.entries(linkSet)) {
			if (strength < MIN_LINK_STRENGTH) continue; // skip weak connections (arbitrary threshold)
			const prevLoc = moveLoc({ ...props, idx: prevIdx });
			const jointLoc = elbow(currLoc, prevLoc);
			const lineStrength = scale(strength, [MIN_LINK_STRENGTH, 1], [255, 0]);
			let color = "";
			if (props.actors.size > 1 && SHOULD_COLORIZE_LINKS) {
				const currActor = props.moves[currIdx].actor || 0;
				const prevActor = props.moves[prevIdx].actor || 0;
				let targetColor = null;
				if (currActor === prevActor && currActor === 0) {
					targetColor = { red: 255, green: 0, blue: 0 };
				}
				else if (currActor === prevActor && currActor === 1) {
					targetColor = { red: 0, green: 0, blue: 255 };
				}
				else {
					targetColor = { red: 160, green: 0, blue: 255 };
				}
				const r = scale(strength, [MIN_LINK_STRENGTH, 1], [255, targetColor.red]);
				const g = scale(strength, [MIN_LINK_STRENGTH, 1], [255, targetColor.green]);
				const b = scale(strength, [MIN_LINK_STRENGTH, 1], [255, targetColor.blue]);
				color = `rgb(${r},${g},${b})`;
			}
			else {
				color = `rgb(${lineStrength},${lineStrength},${lineStrength})`;
			}
			linkLines.push({
				x1: currLoc.x, y1: currLoc.y, x2: jointLoc.x, y2: jointLoc.y, color, strength,
			});
			linkLines.push({
				x1: prevLoc.x, y1: prevLoc.y, x2: jointLoc.x, y2: jointLoc.y, color, strength,
			});
			linkJoints.push({ x: jointLoc.x, y: jointLoc.y, color, strength });
		}
	}
	return { linkLines, linkJoints };
}

function FuzzyLinkograph(props) {
	const dividers = makeTimelineDividers(props);
	const { linkLines, linkJoints } = makeLinkObjects(props);
	return e("div", { className: "fuzzy-linkograph" },
		e("h2", {}, props.title),
		e("svg", { viewBox: `0 0 ${GRAPH_WIDTH} ${(GRAPH_WIDTH / 2) + INIT_Y}` },
			...dividers,
			...linkLines.sort((a, b) => a.strength - b.strength).map(line => {
				return e("line", {
					x1: line.x1, y1: line.y1, x2: line.x2, y2: line.y2,
					stroke: line.color, strokeWidth: 2
				});
			}),
			...linkJoints.sort((a, b) => a.strength - b.strength).map(joint => {
				return e("circle", { cx: joint.x, cy: joint.y, r: 3, fill: joint.color });
			}),
			...props.moves.map((_, idx) => e(DesignMove, { ...props, idx }))
		)
	);
}

/// top-level app init

window.appState = {
	episodes: []
};

let root = null;
function renderUI() {
	if (!root) {
		root = ReactDOM.createRoot(document.getElementById('app'));
	}
	root.render(e("div", {},
		...window.appState.episodes.map(episode => e(FuzzyLinkograph, episode))
	));
}

async function loadDataset(datasetPath) {
	try {
		const json = await (await fetch(datasetPath)).json();
		for (const episodeID of Object.keys(json)) {
			const moves = json[episodeID].moves || json[episodeID]; // if no .moves, assume whole thing is moves list
			const sampleRate = 1; // 30 / moves.length; // downsample to 30ish moves at most
			window.appState.episodes.push({
				title: json[episodeID].title || episodeID,
				moves: moves.filter(x => Math.random() < sampleRate),
				links: json[episodeID].links,
			});
		}
	}
	catch (err) {
		console.log("Couldn't fetch dataset", datasetPath, err);
	}
}

async function main() {
	// load json-formatted data
	// load json-formatted data
	const datasetPaths = window.DATASETS_TO_LOAD || [
		"./data/test.json",
		// "./data/papers.json",
		// "./data/imggen_50.json",
	];
	for (const datasetPath of datasetPaths) {
		await loadDataset(datasetPath);
	}
	// generate linkographs for all episodes
	for (const episode of appState.episodes) {
		episode.actors = new Set(episode.moves.map(m => m.actor || 0));
		if (!episode.links) {
			episode.links = await computeLinks(episode.moves);
		}
		computeLinkDensityIndex(episode);
		computeMoveWeights(episode);
		computeEntropy(episode);
		if (episode.actors.size > 1) {
			computeActorLinkStats(episode);
		}
		episode.moveSpacing = (episode.moves.length > 1)
			? (GRAPH_WIDTH - (INIT_X * 2)) / (episode.moves.length - 1)
			: 0;
		console.log(episode);
	}
	renderUI();
}

main();
