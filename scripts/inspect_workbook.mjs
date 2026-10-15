import fs from 'node:fs/promises';
import {Workbook,SpreadsheetFile,FileBlob} from '@oai/artifact-tool';
const w=await SpreadsheetFile.importXlsx(await FileBlob.load('outputs/face-dataset-v1/얼굴형태_데이터셋_검수.xlsx'));
console.log(w.help('worksheet.images.add',{include:'index,notes,examples',maxChars:5000}).ndjson);
console.log((await w.inspect({kind:'drawing',sheetId:'이미지 목록',maxChars:2500,options:{maxResults:2}})).ndjson);
const b=await w.render({sheetName:'이미지 목록',range:'A1:C5',scale:1.5,format:'png'});
await fs.writeFile('outputs/face-dataset-v1/images_roundtrip.png',new Uint8Array(await b.arrayBuffer()));
