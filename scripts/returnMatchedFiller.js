import fillerMap from '../utils/fillerMap.js';
import {
    loadIndexFromFile,
    vectorSearch,
    getFillerID,
} from '../utils/functions.js';

// Load data to build the indexing
const indexing = await loadIndexFromFile('./index.hnsw', DEBUG);

async function returnMatchedFiller(text, nearestNeighbors) {
    const DEBUG = true;

    // Pass the text into the vectorSearch function
    let result = await vectorSearch(
        text,
        model,
        indexing,
        nearestNeighbors,
        DEBUG
    );

    // Get the fillerText associated with the search result
    let start = performance.now();
    const fillers = result.neighbors.map((id) => {
        return fillerMap.get(getFillerID(id));
    });

    // Add the fillerText to the 'result' object
    const resultWithFiller = { ...result, fillers };

    // Log the performance
    console.table(result);
    if (DEBUG) {
        console.log(
            `\nSearching post processing took ${
                performance.now() - start
            } milliseconds (ie. converting embedding ID into fillerText value).`
        );
    }

    return resultWithFiller;
}

const match = await returnMatchedFiller('Test', 1);
const matchedFiller = match.fillers;

console.log(matchedFiller);
