import fs from 'fs/promises';
import {
    loadModel,
    readAllFilesContent,
    convertToEmbedding,
    buildIndexing,
} from '../utils/functions.js';

async function ingestData() {
    const DEBUG = true;
    // Load data to build the indexing

    const numDimensions = 512; // the length of data point vector that will be indexed.
    const maxElements = 10000; // the maximum number of data points.

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

    const indexPath = './index.hnsw';

    // Save index to file
    await fs.writeFile(
        indexPath,
        JSON
            .stringify
            // indexing.saveIndex()
            // MODIFY TO MAKE IT LEGITIMATE
            ()
    );

    if (DEBUG) {
        console.log('Index saved to:', indexPath);
    }
}

const [match] = await ingestData();
console.log(match);
