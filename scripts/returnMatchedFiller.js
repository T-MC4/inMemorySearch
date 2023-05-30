import fillerMap from '../utils/fillerMap.js';
import {
    loadModel,
    readAllFilesContent,
    convertToEmbedding,
    buildIndexing,
    vectorSearch,
    getFillerID,
} from '../utils/functions.js';

async function returnMatchedFiller(text, top_k) {
    const DEBUG = true;

    // -------------------------------
    // Load data to build the indexing
    const numDimensions = 512; // the length of data point vector that will be indexed.
    const maxElements = 10000; // the maximum number of data points.
    const nearestNeighbors = top_k;

    const model = await loadModel(DEBUG);
    const { ids, contents } = await readAllFilesContent(
        './data/to_process',
        'json',
        DEBUG
    );
    const embeddings = await convertToEmbedding(model, contents, DEBUG);

    const indexing = buildIndexing(
        numDimensions,
        maxElements,
        embeddings,
        ids,
        DEBUG
    );
    //--------------------------------
    // Pass the text into the vectorSearch function
    let result = await vectorSearch(
        text,
        model,
        indexing,
        nearestNeighbors,
        DEBUG
    );
    //--------------------------------

    // Get the fillerText associated with the search result
    let start = performance.now();
    const fillers = result.neighbors.map((id) => {
        return fillerMap.get(getFillerID(id));
    });
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
console.log(match.fillers);
