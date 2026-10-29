/* Paperform — local, coordinate-preserving document editor. */
'use strict';
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const ICONS = {
  inbox:'<path d="M4 4h16v16H4z"/><path d="M4 14h5l2 3h2l2-3h5M12 6v6m-3-3 3 3 3-3"/>',
  layers:'<path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
  export:'<path d="M14 3h7v7m0-7-10 10M10 4H4v16h16v-6"/>',
  shield:'<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z"/><path d="m8 12 3 3 5-6"/>',
  plus:'<path d="M12 5v14M5 12h14"/>',minus:'<path d="M5 12h14"/>',
  'arrow-right':'<path d="M4 12h15m-6-6 6 6-6 6"/>','arrow-left':'<path d="M20 12H5m6-6-6 6 6 6"/>',
  layout:'<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 9v12"/>',
  cursor:'<path d="m5 3 14 10-7 1-3 7-4-18Z"/>',
  code:'<path d="m7 6-5 6 5 6m10-12 5 6-5 6M14 3l-4 18"/>',
  folder:'<path d="M3 7V4h6l3 3h9v13H3V7Z"/>',refresh:'<path d="M20 9a8 8 0 0 0-14-4L3 8m0-5v5h5M4 15a8 8 0 0 0 14 4l3-3m0 5v-5h-5"/>',
  file:'<path d="M5 3h9l5 5v13H5zM14 3v6h5M9 13h6m-6 4h6"/>',
  scan:'<path d="M3 8V3h5m8 0h5v5M3 16v5h5m8 0h5v-5M2 12h20M8 8h8M8 16h8"/>',
  undo:'<path d="M4 4v6h6M4 10a8 8 0 1 1 0 6"/>',redo:'<path d="M20 4v6h-6m6 0a8 8 0 1 0 0 6"/>',
  text:'<path d="M4 5h16M12 5v15M8 20h8M4 5v4m16-4v4"/>',square:'<rect x="4" y="4" width="16" height="16" rx="1"/>',
  image:'<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8" cy="8" r="1.5"/><path d="m3 17 6-6 4 4 3-3 5 5"/>',
  compare:'<path d="M12 2v20M8 4H3v16h5m8-16h5v16h-5M8 8l-3 4 3 4m8-8 3 4-3 4"/>',
  info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
  close:'<path d="m6 6 12 12M6 18 18 6"/>',download:'<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
  eye:'<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
  'eye-off':'<path d="m3 3 18 18M10 5h2c6 0 10 7 10 7a20 20 0 0 1-4 4M6 6a22 22 0 0 0-4 6s4 7 10 7c2 0 3-1 4-1M10 10l4 4"/>',
  lock:'<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3M12 14v3"/>',
  unlock:'<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0M12 14v3"/>',
  copy:'<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V3H3v13h5"/>',
  trash:'<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',
  'arrow-up':'<path d="m6 10 6-6 6 6M12 4v16"/>','arrow-down':'<path d="m6 14 6 6 6-6M12 4v16"/>',
  check:'<path d="m5 12 5 5L20 7"/>'
};
function icon(name){return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ICONS.square}</svg>`;}
function icons(root=document){$$('[data-icon]',root).forEach(node=>{node.innerHTML=icon(node.dataset.icon);});}
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clone = value => JSON.parse(JSON.stringify(value));
const uid = () => 'el_' + (crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36)+Math.random().toString(36).slice(2));
const TYPE_NAMES = {text:'文字',image:'图片',vector:'矢量',region:'图案',background:'背景',shape:'图形'};
const TYPE_ICONS = {text:'text',image:'image',vector:'layers',region:'scan',background:'layout',shape:'square'};
const FALLBACK_FONT_FAMILIES = ['Arial','Microsoft YaHei','SimSun','SimHei','Times New Roman','Georgia','Consolas'];
let fontFamilies = [...FALLBACK_FONT_FAMILIES];
function fontOptions(){return fontFamilies.map(family=>`<option>${escapeHTML(family)}</option>`).join('');}
async function loadFontFamilies(){
  try{
    const data=await(await api('/api/fonts')).json();
    if(!Array.isArray(data?.fonts))return;
    const families=[...FALLBACK_FONT_FAMILIES],seen=new Set(families.map(family=>family.toLowerCase()));
    for(const font of data.fonts){
      const family=typeof font?.family==='string'?font.family.trim():'';
      if(!family||seen.has(family.toLowerCase()))continue;
      seen.add(family.toLowerCase());families.push(family);
    }
    fontFamilies=families;
    const options=$('#font-options');
    if(options)options.innerHTML=fontOptions();
  }catch{/* Font suggestions are optional; the built-in list stays available. */}
}
const IMPORT_MODES = {
  detection:{label:'单步检测预览',action:'检测并打开编辑器',accept:'.pdf',detection:true,description:'仅运行所选检测模型，在原页上显示边框和类别，不识别文字内容、不重排页面。支持 PDF；留空页码处理全部页面，每次最多100页。'},
  html:{label:'PDF → HTML（阅读）',action:'转换并下载 HTML ZIP',accept:'.pdf',reading:true,description:'每次可选择 1–30 份 PDF。完成后直接下载 ZIP，包含阅读型 HTML、图片资源与源 PDF。'},
  md:{label:'PDF → MD（阅读）',action:'转换并下载 Markdown ZIP',accept:'.pdf',reading:true,description:'每次可选择 1–30 份 PDF。完成后直接下载 ZIP，包含阅读型 Markdown、图片资源与源 PDF。'},
  editor:{label:'OCR 解析与画布编辑',action:'解析并打开编辑器',accept:'.pdf,.png,.jpg,.jpeg,.tif,.tiff,.webp,.bmp,.json',description:'原生 PDF 直接提取；扫描内容使用本地版面、文字和表格识别。可恢复项目 JSON，并在画布校对文字、图片与布局。'},
  text_pdf:{label:'PDF / 图片 → 结构化重建',action:'重建并下载',accept:'.pdf,.png,.jpg,.jpeg,.tif,.tiff,.webp,.bmp',forceOCR:true,description:'按类别解析文字、表格和图片，先调整块内结构，再依据阅读关系自适应重排与分页。支持 PDF、PNG、JPEG、多页 TIFF 等；图片及不可靠结构保留原图，输出结果需校对。'}
};
const state = {doc:null,page:0,selected:null,zoom:1,screen:'input',leftTab:'pages',compare:false,files:[],importMode:'editor',optionalOCR:true,readingResult:null,history:[],future:[],revision:0,savedRevision:0,saveTimer:null,saving:null,saveError:null,clipboard:null,drag:null,editing:false,busy:false};
const operationProgress = new OperationProgress();
let toastTimer;
function toast(message,error=false){const node=$('#toast');node.textContent=message;node.classList.toggle('error',error);node.classList.remove('hidden');clearTimeout(toastTimer);toastTimer=setTimeout(()=>node.classList.add('hidden'),error?6500:3000);}
async function api(url,options={}){const response=await operationProgress.request(url,options);if(!response.ok){let detail;try{const body=await response.json();detail=body.detail || body.message;}catch{}throw new Error(typeof detail==='string'?detail:(Array.isArray(detail)?detail.map(x=>x.msg).join('；'):`请求失败 (${response.status})`));}return response;}
function busy(on,title='正在处理你的文档',message='识别文字、保留复杂图文区域，处理时间取决于页面内容。'){if(on)operationProgress.start();else operationProgress.stop();state.busy=on;$('#busy-overlay').classList.toggle('hidden',!on);$('#busy-title').textContent=title;$('#busy-message').textContent=message;$('.app').inert=on;$('.rail').inert=on;$('#input-screen').setAttribute('aria-busy',String(on));updateImportControls();}
function currentPage(){return state.doc?.pages[state.page];}
function selected(){return currentPage()?.elements.find(e=>e.id===state.selected);}
function docStats(){const elements=state.doc?.pages.flatMap(p=>p.elements)||[];return{pages:state.doc?.pages.length||0,elements:elements.length,text:elements.filter(e=>e.type==='text').length,review:elements.filter(e=>e.reviewRequired).length,counts:elements.reduce((a,e)=>(a[e.type]=(a[e.type]||0)+1,a),{})};}
function snapshot(){return JSON.stringify(state.doc);}
function saveLabel(text){$('#save-state').textContent=text;$('#sync-label').textContent=text==='保存失败'?'本地更改尚未写入，请重试':'画布与 JSON 实时同步';}
function refreshStats(){if(!state.doc)return;const stats=docStats();state.doc.stats={...(state.doc.stats||{}),pageCount:stats.pages,elementCount:stats.elements,types:stats.counts,reviewCount:stats.review};state.doc.pages.forEach(page=>page.elements.forEach(el=>{const cx=el.x+el.width/2,cy=el.y+el.height/2,a=(el.rotation||0)*Math.PI/180,c=Math.cos(a),s=Math.sin(a);el.corners=[[-el.width/2,-el.height/2],[el.width/2,-el.height/2],[el.width/2,el.height/2],[-el.width/2,el.height/2]].map(([x,y])=>[round(cx+x*c-y*s),round(cy+x*s+y*c)]);}));}
function commit(before,{redraw=true}={}){if(before===snapshot())return;state.history.push(before);if(state.history.length>70)state.history.shift();state.future=[];state.revision++;state.doc.updatedAt=new Date().toISOString();refreshStats();state.saveError=null;saveLabel('待保存…');scheduleSave();if(redraw)renderEditor(false);else updateHistory();}
function mutate(fn,options){if(!state.doc || state.compare)return;const before=snapshot();fn();commit(before,options);}
function scheduleSave(){clearTimeout(state.saveTimer);state.saveTimer=setTimeout(()=>flushSave().catch(err=>toast(err.message,true)),650);}
async function flushSave(){clearTimeout(state.saveTimer);if(!state.doc)return;if(state.saving){await state.saving;if(state.savedRevision<state.revision)return flushSave();return;}if(state.savedRevision===state.revision)return;const docId=state.doc.id,rev=state.revision,body=snapshot();saveLabel('保存中…');state.saving=(async()=>{try{await api(`/api/documents/${encodeURIComponent(docId)}`,{method:'PUT',headers:{'Content-Type':'application/json'},body});if(state.doc?.id===docId){state.savedRevision=rev;state.saveError=null;saveLabel(state.revision===rev?'已保存到本机':'待保存…');}}catch(error){state.saveError=error;saveLabel('保存失败');throw error;}finally{state.saving=null;}})();await state.saving;if(state.doc?.id===docId && state.revision>rev)return flushSave();}
function setDocument(doc){state.doc=doc;refreshStats();state.page=0;state.selected=null;state.history=[];state.future=[];state.revision=0;state.savedRevision=0;state.compare=false;state.saveError=null;saveLabel('已保存到本机');showScreen('editor');requestAnimationFrame(()=>fitZoom());refreshDocuments();}
function showScreen(screen){if(screen!=='input'&&!state.doc){toast('先导入一份文档，或打开示例体验');return;}if(state.editing)document.activeElement?.blur();state.screen=screen;$$('.screen').forEach(node=>node.classList.add('hidden'));$(`#${screen}-screen`).classList.remove('hidden');$$('[data-screen]').forEach(node=>node.classList.toggle('active',node.dataset.screen===screen));$('#screen-label').textContent={input:'文件工作台',editor:'页面编辑器',output:'成果输出'}[screen];if(screen==='editor'){renderEditor();requestAnimationFrame(()=>renderPage());}if(screen==='output'){renderOutput();flushSave().catch(err=>toast(err.message,true));}}
async function refreshDocuments(){try{const docs=await(await api('/api/documents')).json();$('#recent-count').textContent=docs.length;const root=$('#recent-documents');root.replaceChildren();if(!docs.length){root.innerHTML=`<div class="empty-recent">${icon('folder')}<span>你的文档会保存在这里，随时回来继续编辑。</span></div>`;return;}docs.forEach(doc=>{const button=document.createElement('button');button.className='recent-card';button.innerHTML=`${icon('file')}<div><strong>${escapeHTML(doc.name)}</strong><small>${doc.pageCount} 页 · ${doc.elementCount} 个元素 · ${dateLabel(doc.updatedAt)}</small></div><span class="arrow">${icon('arrow-right')}</span>`;button.onclick=()=>openDocument(doc.id);root.append(button);});}catch(error){$('#recent-documents').innerHTML=`<div class="empty-recent">${icon('info')}<span>${escapeHTML(error.message)}，请检查本地服务后刷新。</span></div>`;}}
function dateLabel(value){const date=new Date(value);return Number.isNaN(date.getTime())?'已保存':date.toLocaleDateString('zh-CN',{month:'short',day:'numeric'});}
async function openDocument(id){try{await flushSave();busy(true,'正在打开文档','载入保存的页面与可编辑元素。');const doc=await(await api(`/api/documents/${encodeURIComponent(id)}`)).json();setDocument(doc);}catch(error){toast(error.message,true);}finally{busy(false);}}
function acceptsFile(file){return (state.importMode==='editor'?/\.(pdf|png|jpe?g|tiff?|webp|bmp|json)$/i:state.importMode==='text_pdf'?/\.(pdf|png|jpe?g|tiff?|webp|bmp)$/i:/\.pdf$/i).test(file.name);}
function getPages(){
  const pages=$('#page-range').value.trim().replace(/，/g,',').replace(/\s*([,-])\s*/g,'$1');
  if(!pages)return '';
  if(!/^[1-9]\d*(?:-[1-9]\d*)?(?:,[1-9]\d*(?:-[1-9]\d*)?)*$/.test(pages))throw new Error('物理页码请填写 1-3,5,8-10 这样的格式，页码从 1 开始。');
  for(const part of pages.split(',')){const [start,end=start]=part.split('-').map(Number);if(!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||end<start)throw new Error('物理页码范围无效：请使用正整数，结束页不能小于起始页。');}
  return pages;
}
function importValidationError(){
  if(state.files.some(file=>!acceptsFile(file)))return state.importMode==='text_pdf'?'结构化重建支持 PDF、PNG、JPG、TIFF、WebP 和 BMP，请移除不支持的文件。':`${IMPORT_MODES[state.importMode].label}仅接受 PDF 文件，请移除不支持的文件，或切回 OCR 解析与画布编辑。`;
  if(state.files.length>30)return '每次可选择 1–30 个文件，请移除多余文件后重试。';
  try{getPages();}catch(error){return error.message;}
  return '';
}
function showImportError(message=''){const node=$('#import-error');node.textContent=message;node.classList.toggle('hidden',!message);}
function updateImportControls(){
  const mode=IMPORT_MODES[state.importMode];
  $$('[data-import-mode]').forEach(input=>input.disabled=state.busy);
  ['file-input','choose-files','page-range','rebuild-format','detection-model'].forEach(id=>$('#'+id).disabled=state.busy);
  $$('#pending-files button').forEach(button=>button.disabled=state.busy);
  $('#ocr-toggle').disabled=state.busy||!!mode.forceOCR||!!mode.detection;
  if(mode.forceOCR)$('#ocr-toggle').checked=true;
  $('#dropzone').setAttribute('aria-disabled',String(state.busy));
  $('#dropzone').tabIndex=state.busy?-1:0;
  $('#parse-files').disabled=state.busy||!state.files.length||!!importValidationError();
  const action=state.importMode==='text_pdf'?`重建并下载 ${$('#rebuild-format').value==='pdf'?'PDF':$('#rebuild-format').value==='html'?'HTML ZIP':'Markdown ZIP'}`:mode.action;
  $('#parse-label').textContent=action+(state.files.length?` · ${state.files.length} 个文件`:'');
  let invalidPages=false;try{getPages();}catch{invalidPages=true;}
  $('#page-range').setAttribute('aria-invalid',String(invalidPages));
}
function setImportMode(value){
  if(state.busy||!IMPORT_MODES[value])return;
  if(!IMPORT_MODES[state.importMode].forceOCR)state.optionalOCR=$('#ocr-toggle').checked;
  state.importMode=value;
  const mode=IMPORT_MODES[value];
  $$('[data-import-mode]').forEach(input=>input.checked=input.value===value);
  $('#file-input').accept=mode.accept;
  $('#file-types-label').textContent=value==='editor'?'PDF / IMAGE / JSON':value==='text_pdf'?'PDF / IMAGE':'PDF ONLY';
  $('#file-formats').innerHTML=value==='editor'?'PDF · PNG · JPG · TIFF · WebP · BMP · JSON <span>支持多文件、多页文档 · JSON 可恢复编辑项目</span>':value==='text_pdf'?'PDF · PNG · JPG · TIFF · WebP · BMP <span>每次可选择 1–30 个文件 · TIFF 支持多页</span>':'PDF <span>每次可选择 1–30 份 PDF · 支持多页文档</span>';
  $('#dropzone').setAttribute('aria-label',value==='editor'?'选择或拖入 PDF、图片和项目 JSON':value==='text_pdf'?'选择或拖入 PDF 和图片文件':'选择或拖入 PDF 文件');
  $('#import-mode-description').textContent=mode.description;
  $('#rebuild-options').classList.toggle('hidden',value!=='text_pdf');
  $('#detection-options').classList.toggle('hidden',value!=='detection');
  $('#ocr-toggle').closest('.upload-footer').classList.toggle('hidden',!!mode.detection);
  $('#ocr-help').classList.toggle('hidden',!!mode.detection);
  $('#ocr-toggle').checked=mode.forceOCR||state.optionalOCR;
  $('#ocr-label').textContent=mode.forceOCR?'图片内容识别（必须开启）':'识别扫描件中的文字';
  $('#ocr-help').textContent=mode.forceOCR?'原生 PDF 文字和表格直接提取；仅图片内容调用 V3，支持全部页面或所选页码。':'扫描内容可启用本地 OCR，识别结果仍需校对。';
  $('#ocr-toggle').closest('.switch-label').classList.toggle('ocr-required',!!mode.forceOCR);
  renderPending();
}
function setFiles(files){
  if(state.busy)return;
  const incoming=[...files],accepted=incoming.filter(acceptsFile);
  state.files.push(...accepted);renderPending();
  if(accepted.length!==incoming.length){const message=state.importMode==='editor'?'支持 PDF、PNG、JPG、TIFF、WebP、BMP 和项目 JSON。':state.importMode==='text_pdf'?'重建支持 PDF、PNG、JPG、TIFF、WebP 和 BMP。':`${IMPORT_MODES[state.importMode].label}仅接受 PDF 文件。`;showImportError(message);toast(message,true);}
}
function renderPending(){
  const root=$('#pending-files');root.replaceChildren();
  state.files.forEach((file,index)=>{const valid=acceptsFile(file),row=document.createElement('div');row.className='pending-file'+(valid?'':' invalid');row.dataset.fileValid=String(valid);row.innerHTML=`${icon(file.name.toLowerCase().endsWith('.pdf')?'file':'image')}<span>${escapeHTML(file.name)}</span><small>${valid?(file.size>1024*1024?(file.size/1024/1024).toFixed(1)+' MB':Math.ceil(file.size/1024)+' KB'):'此模式仅接受 PDF'}</small><button class="icon-button" aria-label="移除文件 ${escapeHTML(file.name)}">${icon('close')}</button>`;$('button',row).onclick=()=>{if(state.busy)return;state.files.splice(index,1);renderPending();};root.append(row);});
  updateImportControls();showImportError(importValidationError());
}
function downloadFilename(response,fallback){
  const disposition=response.headers.get('content-disposition')||'';
  const encoded=/filename\*\s*=\s*UTF-8'[^']*'([^;]+)/i.exec(disposition);
  let filename;
  if(encoded){try{filename=decodeURIComponent(encoded[1].trim());}catch{}}
  if(!filename){const match=/filename\s*=\s*(?:"((?:\\.|[^"\\])*)"|([^;]+))/i.exec(disposition);filename=match&&(match[1]!=null?match[1].replace(/\\(.)/g,'$1'):match[2].trim());}
  return (filename||fallback).replace(/[\\/:*?"<>|\u0000-\u001f\u007f]/g,'_');
}
function triggerDownload(url,filename){const link=document.createElement('a');link.href=url;link.download=filename;document.body.append(link);link.click();link.remove();}
function releaseReadingResult(){
  if(state.readingResult)URL.revokeObjectURL(state.readingResult.url);
  state.readingResult=null;
  $('#reading-download').removeAttribute('href');$('#reading-download').removeAttribute('download');
  $('#reading-result').classList.add('hidden');
}
async function completeReadingDownload(response,format,count,pages,rebuild=false){
  const directPDF=format==='text_pdf';
  const blob=await operationProgress.download(response),filename=downloadFilename(response,directPDF?'text-document.pdf':`reading-${format}.zip`),url=URL.createObjectURL(blob);
  releaseReadingResult();
  state.readingResult={url,filename,format,count,pages};
  const label=format==='html'?'HTML':'Markdown';
  $('#reading-result').dataset.format=format;
  $('#reading-result').dataset.downloadType=directPDF?'pdf':'zip';
  $('#reading-result-title').textContent=rebuild?`${directPDF?'PDF':label} 重建完成 · ${count} 份输入文件`:directPDF?`可编辑文本 PDF 转换完成 · ${count} 份输入 PDF`:`${label} 转换完成 · ${count} 份 PDF`;
  const description=rebuild?`已生成${directPDF?'可编辑 PDF':`${label} ZIP`}。文字、表格和图片按块测量后，以单栏阅读顺序自适应排版、分页；OCR、表格结构和选用样式需校对。${response.headers.get('x-reconstruction-status')==='partial'?'部分区域保留为图片，不能独立编辑其中的文字。':''}`:directPDF?'已生成可编辑文本 PDF。正文为可编辑、可搜索、可选择的真实文字，复杂图文区域整体保留为图片；OCR 结果仍需核对。':`ZIP 包含 ${label} 文档、图片资源与源 PDF。`;
  $('#reading-result-description').textContent=`${description}${pages?'所选物理页码：'+pages+'。':'已处理全部页面。'}已发起下载，也可使用下方链接重新下载。`;
  $('#reading-download-label').textContent=directPDF?'重新下载 PDF':'重新下载 ZIP';
  $('#reading-download').href=url;$('#reading-download').download=filename;
  $('#reading-result').classList.remove('hidden');
  triggerDownload(url,filename);
}
async function directTextPDF(){
  if(state.busy||state.importMode!=='text_pdf'||!state.files.length)return;
  const errorMessage=importValidationError();if(errorMessage){showImportError(errorMessage);return;}
  const files=[...state.files],pages=getPages(),format=$('#rebuild-format').value;
  showImportError();busy(true,'正在自适应重建文档','按类别解析文字、表格和图片，测量块内尺寸，再调整块间位置与分页；结果仍需校对。');
  try{
    const data=new FormData();files.forEach(file=>data.append('files',file));data.append('ocr','true');data.append('pages',pages);
    data.append('format',format);
    const response=await api('/api/convert/rebuild',{method:'POST',body:data});
    await completeReadingDownload(response,format==='pdf'?'text_pdf':format,files.length,pages,true);
    state.files=[];renderPending();toast('重建结果已生成，请对照原稿校对');
  }catch(error){showImportError(error.message);toast(error.message,true);}finally{busy(false);}
}
async function parseFiles(){
  if(state.busy||!state.files.length)return;
  if(state.importMode==='detection')return detectFiles();
  if(state.importMode==='text_pdf')return directTextPDF();
  const errorMessage=importValidationError();if(errorMessage){showImportError(errorMessage);return;}
  const modeName=state.importMode,mode=IMPORT_MODES[modeName],files=[...state.files],pages=getPages();
  const ocr=!!mode.forceOCR||$('#ocr-toggle').checked;
  showImportError();busy(true,`正在${mode.label}`,mode.reading?'正在生成阅读文档、图片资源与源 PDF 的 ZIP 包。':'识别文字并保留复杂图文区域，完成后打开页面编辑器。');
  try{
    const data=new FormData();files.forEach(file=>data.append('files',file));data.append('ocr',String(ocr));data.append('pages',pages);
    if(mode.reading){
      data.append('format',modeName);
      const response=await api('/api/convert/reading',{method:'POST',body:data});
      await completeReadingDownload(response,modeName,files.length,pages);
      toast(`${modeName==='html'?'HTML':'Markdown'} ZIP 已生成，可重新下载`);
    }else{
      await flushSave();data.append('mode','editable');
      const doc=await(await api('/api/documents',{method:'POST',body:data})).json();
      await operationProgress.complete('解析完成','文档数据已接收，正在打开页面编辑器。');
      setDocument(doc);
      toast(`已解析 ${doc.pages.length} 页文档`);
    }
    state.files=[];renderPending();
  }catch(error){showImportError(error.message);toast(error.message,true);}finally{busy(false);}
}
async function detectFiles(){
  const errorMessage=importValidationError();if(errorMessage){showImportError(errorMessage);return;}
  const files=[...state.files],pages=getPages(),detector=$('#detection-model').value;
  showImportError();busy(true,'正在生成检测预览','仅运行所选模型，保留原页并绘制检测边框；完成后打开页面编辑器。');
  try{
    await flushSave();
    const data=new FormData();files.forEach(file=>data.append('files',file));
    data.append('pages',pages);data.append('detector',detector);
    const doc=await(await api('/api/detection-preview',{method:'POST',body:data})).json();
    await operationProgress.complete('检测完成','边框与类别已接收，正在打开页面编辑器。');
    setDocument(doc);state.files=[];renderPending();toast('检测完成，可在图层中选择或隐藏边框与类别');
  }catch(error){showImportError(error.message);toast(error.message,true);}finally{busy(false);}
}
async function loadDemo(){try{await flushSave();busy(true,'正在准备示例文档','打开包含文字、矢量图形与图片的可编辑示例。');const doc=await(await api('/api/demo',{method:'POST'})).json();await operationProgress.complete('示例已生成','文档数据已接收，正在打开页面编辑器。');setDocument(doc);toast('示例已打开，试着点击并拖动页面上的元素');}catch(error){toast(error.message,true);}finally{busy(false);}}
function renderEditor(rebuild=true){if(!state.doc)return;const stats=docStats();$('#document-name').value=state.doc.name;$('#document-meta').textContent=`${stats.pages} 页 · ${stats.elements} 个独立元素${stats.review?' · '+stats.review+' 项建议复核':''}`;$('#layer-count').textContent=currentPage()?.elements.length||0;$('#page-summary').textContent=`第 ${state.page+1} / ${stats.pages} 页 · pt 坐标`;$('#canvas-page-name').textContent=`页面 ${String(state.page+1).padStart(2,'0')}  /  ${currentPage()?.width.toFixed(0)} × ${currentPage()?.height.toFixed(0)} pt`;$('#compare-toggle').setAttribute('aria-pressed',String(state.compare));$('#compare-label').classList.toggle('hidden',!state.compare);renderPages();renderLayers();renderProperties();renderPage();updateHistory();}
function updateHistory(){$('#undo').disabled=!state.history.length||state.compare;$('#redo').disabled=!state.future.length||state.compare;['add-text','add-shape','add-image'].forEach(id=>$('#'+id).disabled=state.compare);}
function renderPages(){const root=$('#pages-list');root.replaceChildren();state.doc.pages.forEach((page,index)=>{const button=document.createElement('button');button.className='page-thumbnail'+(state.page===index?' active':'');button.innerHTML=`<span class="thumbnail-image"><img alt="第 ${index+1} 页原始缩略图" loading="lazy" src="${escapeHTML(safeSrc(page.sourceSrc))}"></span><small>${String(index+1).padStart(2,'0')} <span>· ${page.elements.length} 元素</span></small>`;button.onclick=()=>{state.page=index;state.selected=null;renderEditor();fitZoom();};root.append(button);});}
function safeSrc(src){return typeof src==='string'&&(/^(data:image\/(png|jpeg|webp|gif|bmp|svg\+xml);|\/files\/|blob:)/i.test(src))?src:'';}
function renderLayers(){const root=$('#layers-list');root.replaceChildren();[...currentPage().elements].sort((a,b)=>b.zIndex-a.zIndex).forEach(el=>{const row=document.createElement('div');row.className='layer-row'+(el.id===state.selected?' selected':'')+(el.visible===false?' invisible':'');row.tabIndex=0;row.title=`${TYPE_NAMES[el.type]||el.type} · ${el.name || el.text || el.id}`;row.innerHTML=`<span class="layer-type">${icon(TYPE_ICONS[el.type]||'square')}</span><span class="layer-name">${escapeHTML(el.name||el.text||TYPE_NAMES[el.type]||el.id)}</span>${el.reviewRequired?'<i class="review-dot" title="建议复核"></i>':''}<button class="icon-button visibility-button" title="${el.visible===false?'显示':'隐藏'}">${icon(el.visible===false?'eye-off':'eye')}</button><button class="icon-button lock-button" title="${el.locked?'解锁':'锁定'}">${icon(el.locked?'lock':'unlock')}</button>`;row.onclick=()=>selectElement(el.id);row.onkeydown=event=>{if(event.key==='Enter')selectElement(el.id);};$('.visibility-button',row).onclick=event=>{event.stopPropagation();mutate(()=>el.visible=el.visible===false);};$('.lock-button',row).onclick=event=>{event.stopPropagation();mutate(()=>el.locked=!el.locked);};root.append(row);});}
function styleElement(node,el){Object.assign(node.style,{left:`${el.x}px`,top:`${el.y}px`,width:`${el.width}px`,height:`${el.height}px`,transform:`rotate(${el.rotation||0}deg)`,zIndex:String(el.zIndex||0),opacity:String(el.opacity??1),display:el.visible===false?'none':''});}
function renderElement(el){const node=document.createElement('div');node.className='canvas-element'+(el.locked?' locked':'');node.dataset.id=el.id;styleElement(node,el);const appearance=safeSrc(el.appearanceSrc);if(appearance){const img=document.createElement('img');img.src=appearance;img.alt='';img.draggable=false;node.append(img);}else if(el.type==='text'){const text=document.createElement('div');text.className='element-text';text.textContent=el.text||'';Object.assign(text.style,{fontFamily:el.fontFamily||'Arial, sans-serif',fontSize:`${el.fontSize||12}px`,fontWeight:String(el.fontWeight||400),fontStyle:el.fontStyle||'normal',color:el.color||'#222222',textAlign:el.align||'left',lineHeight:String(el.lineHeight||1.2)});node.append(text);}else if(el.type==='vector'&&el.svg){const img=document.createElement('img');img.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(styledSvg(el));img.alt='';img.draggable=false;node.append(img);}else if(el.type==='shape'){node.innerHTML=shapeSvg(el);}else{const img=document.createElement('img');img.src=safeSrc(el.src);img.alt='';img.draggable=false;node.append(img);}node.onpointerdown=event=>startDrag(event,el);node.ondblclick=event=>{event.stopPropagation();if(el.type==='text')beginTextEdit(el);};return node;}
function shapeSvg(el){const w=Math.max(1,el.width),h=Math.max(1,el.height),sw=Math.max(0,el.strokeWidth??1),fill=escapeHTML(el.fill||'none'),stroke=escapeHTML(el.stroke||'none');let inner;if(el.shape==='ellipse')inner=`<ellipse cx="${w/2}" cy="${h/2}" rx="${Math.max(0,(w-sw)/2)}" ry="${Math.max(0,(h-sw)/2)}"/>`;else if(el.shape==='line')inner=`<line x1="0" y1="${h/2}" x2="${w}" y2="${h/2}"/>`;else inner=`<rect x="${sw/2}" y="${sw/2}" width="${Math.max(0,w-sw)}" height="${Math.max(0,h-sw)}"/>`;return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}">${inner}</svg>`;}
function renderPage(){const page=currentPage();if(!page)return;const canvas=$('#page-canvas'),stage=$('#canvas-stage');canvas.replaceChildren();Object.assign(canvas.style,{width:`${page.width}px`,height:`${page.height}px`,transform:`scale(${state.zoom})`});Object.assign(stage.style,{width:`${page.width*state.zoom}px`,height:`${page.height*state.zoom}px`});$('#zoom-fit').textContent=Math.round(state.zoom*100)+'%';if(state.compare){const img=document.createElement('img');img.className='original-page';img.src=safeSrc(page.sourceSrc);img.alt='原始页面';canvas.append(img);return;}[...page.elements].sort((a,b)=>a.zIndex-b.zIndex).forEach(el=>canvas.append(renderElement(el)));renderSelection();}
function renderSelection(){const canvas=$('#page-canvas');$('.selection-box',canvas)?.remove();const el=selected();if(!el||el.visible===false||state.compare||state.editing)return;const box=document.createElement('div');box.className='selection-box';styleElement(box,{...el,opacity:1,zIndex:2147483000});box.innerHTML=`<span class="selection-label">${escapeHTML(TYPE_NAMES[el.type]||'元素')}${el.locked?' · 已锁定':''}</span>`;if(!el.locked){box.innerHTML+='<i class="selection-handle corner-tl"></i><i class="selection-handle corner-tr"></i><i class="selection-handle corner-bl"></i><i class="selection-handle resize"></i><i class="rotation-line"></i><i class="selection-handle rotate"></i>';$('.resize',box).onpointerdown=event=>startTransform(event,'resize');$('.rotate',box).onpointerdown=event=>startTransform(event,'rotate');}canvas.append(box);}
function selectElement(id){if(state.compare)return;if(state.editing)document.activeElement?.blur();state.selected=id;renderSelection();renderLayers();renderProperties();}
function fitZoom(){const page=currentPage(),wrap=$('#canvas-wrap');if(!page||!wrap.clientWidth)return;state.zoom=Math.min(1.4,Math.max(.1,Math.min((wrap.clientWidth-70)/page.width,(wrap.clientHeight-85)/page.height)));renderPage();}
function setZoom(value){state.zoom=Math.min(3,Math.max(.1,value));renderPage();}
function startDrag(event,el){if(event.button!==0||state.compare||state.editing)return;event.preventDefault();event.stopPropagation();selectElement(el.id);if(el.locked)return;state.drag={kind:'move',id:el.id,startX:event.clientX,startY:event.clientY,original:clone(el),before:snapshot()};document.addEventListener('pointermove',pointerMove);document.addEventListener('pointerup',pointerUp,{once:true});}
function startTransform(event,kind){event.preventDefault();event.stopPropagation();const el=selected();if(!el||el.locked)return;const pageRect=$('#page-canvas').getBoundingClientRect();const center={x:pageRect.left+(el.x+el.width/2)*state.zoom,y:pageRect.top+(el.y+el.height/2)*state.zoom};state.drag={kind,id:el.id,startX:event.clientX,startY:event.clientY,original:clone(el),before:snapshot(),center,startAngle:Math.atan2(event.clientY-center.y,event.clientX-center.x)};document.addEventListener('pointermove',pointerMove);document.addEventListener('pointerup',pointerUp,{once:true});}
function pointerMove(event){const drag=state.drag;if(!drag)return;const el=currentPage().elements.find(e=>e.id===drag.id);if(!el)return;const dx=(event.clientX-drag.startX)/state.zoom,dy=(event.clientY-drag.startY)/state.zoom,o=drag.original;if(drag.kind==='move'){el.x=Math.max(-50000,Math.min(50000,round(o.x+dx)));el.y=Math.max(-50000,Math.min(50000,round(o.y+dy)));}else if(drag.kind==='resize'){const angle=(o.rotation||0)*Math.PI/180;let dw=dx*Math.cos(angle)+dy*Math.sin(angle),dh=-dx*Math.sin(angle)+dy*Math.cos(angle);if(event.shiftKey){dh=dw*o.height/o.width;}const w=Math.min(20000,Math.max(2,o.width+dw)),h=Math.min(20000,Math.max(2,o.height+dh));dw=w-o.width;dh=h-o.height;el.width=round(w);el.height=round(h);el.x=round(o.x+(dw*Math.cos(angle)-dh*Math.sin(angle)-dw)/2);el.y=round(o.y+(dw*Math.sin(angle)+dh*Math.cos(angle)-dh)/2);}else if(drag.kind==='rotate'){const angle=Math.atan2(event.clientY-drag.center.y,event.clientX-drag.center.x);let rotation=(o.rotation||0)+(angle-drag.startAngle)*180/Math.PI;if(event.shiftKey)rotation=Math.round(rotation/15)*15;el.rotation=round(((rotation+180)%360+360)%360-180);}const node=$$('.canvas-element').find(n=>n.dataset.id===el.id);if(node){styleElement(node,el);if(el.type==='shape')node.innerHTML=shapeSvg(el);}renderSelection();['x','y','width','height','rotation'].forEach(key=>{const input=$(`[data-prop="${key}"]`);if(input)input.value=round(el[key]||0);});}
function pointerUp(){document.removeEventListener('pointermove',pointerMove);const drag=state.drag;state.drag=null;if(drag)commit(drag.before);}
function round(value){return Math.round(value*100)/100;}
function beginTextEdit(el){if(el.locked||state.compare)return;selectElement(el.id);const before=snapshot();const original=clone(el);if(el.appearanceSrc)el.originalAppearanceSrc=el.originalAppearanceSrc||el.appearanceSrc;delete el.appearanceSrc;let cancelled=false;state.editing=true;renderPage();const node=$$('.canvas-element').find(n=>n.dataset.id===el.id);const text=$('.element-text',node);text.contentEditable='true';text.style.opacity='1';text.focus();const range=document.createRange();range.selectNodeContents(text);const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);text.onpointerdown=event=>event.stopPropagation();text.onkeydown=event=>{event.stopPropagation();if(event.key==='Escape'){event.preventDefault();cancelled=true;text.blur();}if((event.ctrlKey||event.metaKey)&&event.key==='Enter'){event.preventDefault();text.blur();}};text.onblur=()=>{if(cancelled||text.innerText===(original.text||'')){Object.keys(el).forEach(key=>delete el[key]);Object.assign(el,original);}else{el.text=text.innerText;}state.editing=false;commit(before);renderEditor(false);};}
function numberField(label,key,value,min,max,step='.1'){const limits={x:[-50000,50000],y:[-50000,50000],width:[.1,20000],height:[.1,20000],rotation:[-360000,360000],fontSize:[1,1000],lineHeight:[.5,5],strokeWidth:[0,1000]};if(limits[key]){min=min??limits[key][0];max=max??limits[key][1];}return `<label class="field">${label}<input type="number" data-prop="${key}" value="${round(value??0)}"${min!==undefined?` min="${min}"`:''}${max!==undefined?` max="${max}"`:''} step="${step}"></label>`;}
function styledSvg(el){const xml=new DOMParser().parseFromString(el.svg,'image/svg+xml');xml.querySelectorAll('path,rect,circle,ellipse,line,polyline,polygon').forEach(node=>{if(el.fill!=null)node.setAttribute('fill',el.fill);if(el.stroke!=null){node.setAttribute('stroke',el.stroke);node.setAttribute('stroke-width',String(el.strokeWidth??1));}});return new XMLSerializer().serializeToString(xml.documentElement);}
function vectorValue(el,key,fallback){if(el[key]!=null)return el[key];const xml=new DOMParser().parseFromString(el.svg,'image/svg+xml');const attr=key==='strokeWidth'?'stroke-width':key;return xml.querySelector(`[${attr}]:not([${attr}="none"])`)?.getAttribute(attr)||fallback;}
function selectField(label,key,value,options){return `<label class="field">${label}<select data-prop="${key}">${options.map(([v,l])=>`<option value="${v}"${String(value)===String(v)?' selected':''}>${l}</option>`).join('')}</select></label>`;}
function colorField(label,key,value){const valid=/^#[0-9a-f]{6}$/i.test(value||'')?value:'#ffffff';return `<label class="field">${label}<input type="color" data-prop="${key}" value="${valid}"></label>`;}
function renderProperties(){const root=$('#properties-content'),el=selected(),page=currentPage();$('#selection-type').textContent=el?(TYPE_NAMES[el.type]||el.type):'页面信息';if(!el){root.innerHTML=`<div class="inspector-empty">${icon('cursor')}<h3>${state.compare?'正在查看原稿':'选择一个元素'}</h3><p>${state.compare?'关闭「查看原稿」后继续编辑。':'点击画布中的内容<br>或在图层列表中选择元素。'}</p></div><div class="document-info"><div class="info-row"><span>页面尺寸</span><strong>${round(page.width)} × ${round(page.height)}</strong></div><div class="info-row"><span>页面元素</span><strong>${page.elements.length} 个</strong></div><div class="info-row"><span>坐标单位</span><strong>pt · 左上角原点</strong></div><div class="type-counts">${Object.entries(page.elements.reduce((a,e)=>(a[e.type]=(a[e.type]||0)+1,a),{})).map(([type,count])=>`<span>${TYPE_NAMES[type]||type} ${count}</span>`).join('')}</div></div>${page.warnings?.length?'<div class="inspector-section">'+page.warnings.slice(0,4).map(w=>`<div class="warning-note">${escapeHTML(w)}</div>`).join('')+'</div>':''}`;return;}
  let html=`<div class="inspector-section"><label class="field">元素名称<input data-prop="name" value="${escapeHTML(el.name||'')}"></label><div class="field-label-row" style="margin-top:12px"><label class="checkbox-field"><input type="checkbox" data-prop="visible"${el.visible!==false?' checked':''}>可见</label><label class="checkbox-field"><input type="checkbox" data-prop="locked"${el.locked?' checked':''}>锁定</label></div></div><div class="inspector-section"><h3>位置与变换 <span class="subtle">pt</span></h3><div class="field-grid">${numberField('X 坐标','x',el.x)}${numberField('Y 坐标','y',el.y)}${numberField('宽度','width',el.width,.1)}${numberField('高度','height',el.height,.1)}${numberField('旋转角度 °','rotation',el.rotation||0)}${numberField('不透明度','opacity',el.opacity??1,0,1,.05)}</div></div>`;
  if(el.type==='text'){html+=`<div class="inspector-section"><h3>文字内容</h3><div class="field-grid"><label class="field full"><textarea data-prop="text">${escapeHTML(el.text||'')}</textarea></label><label class="field full">字体<input data-prop="fontFamily" list="font-options" value="${escapeHTML(el.fontFamily||'Arial')}"><datalist id="font-options">${fontOptions()}</datalist></label>${numberField('字号','fontSize',el.fontSize||12,1,1000)}${colorField('文字颜色','color',el.color)}${selectField('字重','fontWeight',el.fontWeight||400,[['normal','常规'],['bold','粗体'],[100,'极细'],[200,'特细'],[300,'细体'],[400,'常规 400'],[500,'中等'],[600,'半粗'],[700,'粗体 700'],[800,'特粗'],[900,'极粗']])}${selectField('字形','fontStyle',el.fontStyle||'normal',[['normal','常规'],['italic','斜体']])}${selectField('对齐','align',el.align||'left',[['left','左对齐'],['center','居中'],['right','右对齐'],['justify','两端对齐']])}${numberField('行高倍数','lineHeight',el.lineHeight||1.2,.5,5)}</div></div>`;}
  if(el.type==='shape'){html+=`<div class="inspector-section"><h3>图形样式</h3><div class="field-grid">${selectField('形状','shape',el.shape||'rect',[['rect','矩形'],['ellipse','椭圆'],['line','直线']])}${numberField('描边宽度','strokeWidth',el.strokeWidth||0,0,100)}${colorField('填充颜色','fill',el.fill)}${colorField('描边颜色','stroke',el.stroke)}</div><label class="checkbox-field" style="margin-top:10px"><input type="checkbox" id="shape-no-fill"${el.fill==='none'?' checked':''}>无填充</label></div>`;}
  if(el.type==='vector'){html+=`<div class="inspector-section"><h3>矢量样式</h3><div class="field-grid">${colorField('填充颜色','fill',vectorValue(el,'fill','#ffffff'))}${colorField('描边颜色','stroke',vectorValue(el,'stroke','#233c47'))}${numberField('描边宽度','strokeWidth',Number(vectorValue(el,'strokeWidth',1)),0,1000)}</div><label class="checkbox-field" style="margin-top:10px"><input type="checkbox" id="shape-no-fill"${el.fill==='none'?' checked':''}>无填充</label><label class="checkbox-field" style="margin-top:8px"><input type="checkbox" id="vector-no-stroke"${el.stroke==='none'?' checked':''}>无描边</label><p class="appearance-note" style="margin-top:10px">颜色修改应用于该矢量图层中的全部路径。</p><button class="inspector-link" id="reset-vector-style">恢复原始矢量样式</button></div>`;}
  if(['image','region','background'].includes(el.type)){html+=`<div class="inspector-section"><h3>图像素材</h3><p class="appearance-note">替换素材保留当前元素的位置、尺寸与旋转角度。</p><button id="replace-image" class="button secondary small">${icon('image')}替换图片</button><input id="replacement-input" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" hidden></div>`;}
  if(el.appearanceSrc||el.originalAppearanceSrc){html+=`<div class="inspector-section"><h3>外观保真</h3><p class="appearance-note">${el.appearanceSrc?'<strong>正在使用原始外观</strong><br>移动与变换保留特殊字形；修改文字或字体后切换为可编辑样式。':'正在显示可编辑样式。可恢复原始图像外观（不会还原文字数据）。'}</p><button class="inspector-link" id="toggle-appearance">${el.appearanceSrc?'显示可编辑样式':'恢复原始外观'}</button></div>`;}
  if(el.reviewRequired||el.confidence!=null){html+=`<div class="inspector-section"><div class="warning-note">${el.reviewRequired?'此元素建议与原稿核对。特殊字形、公式或低质量扫描可能影响识别。':''}${el.confidence!=null?'<br>识别置信度：'+Math.round(el.confidence*100)+'%':''}</div>${el.reviewRequired?'<button id="mark-reviewed" class="inspector-link" style="margin-top:10px">标记为已复核</button>':''}</div>`;}
  html+=`<div class="inspector-section"><h3>图层操作 <span class="subtle">${el.zIndex}</span></h3><div class="inspector-actions"><button class="button secondary" id="layer-forward">${icon('arrow-up')}上移</button><button class="button secondary" id="layer-backward">${icon('arrow-down')}下移</button><button class="button secondary" id="duplicate-element">${icon('copy')}复制</button><button class="button secondary danger" id="delete-element">${icon('trash')}删除</button></div></div>`;
  root.innerHTML=html;$$('[data-prop]',root).forEach(input=>{if(el.locked&&!['locked','visible'].includes(input.dataset.prop))input.disabled=true;const live=input.type!=='checkbox'&&input.tagName!=='SELECT';input.onfocus=()=>{input._before=snapshot();input._grouped=false;};input.oninput=()=>{if(live)updateProperty(el,input,true);};input.onchange=()=>updateProperty(el,input,live);input.onblur=()=>{input._before=null;input._grouped=false;};});$('#shape-no-fill')?.addEventListener('change',event=>mutate(()=>{el.fill=event.target.checked?'none':'#dce9d7';delete el.appearanceSrc;}));$('#replace-image')?.addEventListener('click',()=>$('#replacement-input').click());$('#replacement-input')?.addEventListener('change',event=>replaceImage(event.target.files[0],el));$('#toggle-appearance')?.addEventListener('click',()=>mutate(()=>{if(el.appearanceSrc){el.originalAppearanceSrc=el.originalAppearanceSrc||el.appearanceSrc;delete el.appearanceSrc;}else if(el.originalAppearanceSrc)el.appearanceSrc=el.originalAppearanceSrc;}));$('#mark-reviewed')?.addEventListener('click',()=>mutate(()=>el.reviewRequired=false));$('#layer-forward').onclick=()=>reorder(1);$('#layer-backward').onclick=()=>reorder(-1);$('#duplicate-element').onclick=duplicateElement;$('#delete-element').onclick=deleteElement;}
function updateProperty(el,input,live=false){
  const key=input.dataset.prop;if(el.locked&&!['locked','visible'].includes(key))return;
  let value=input.type==='checkbox'?input.checked:input.value;
  const numeric=['x','y','width','height','rotation','opacity','fontSize','lineHeight','strokeWidth'];
  if(numeric.includes(key)){if(input.value==='')return;value=Number(value);if(!Number.isFinite(value))return;if(input.min!=='')value=Math.max(Number(input.min),value);if(input.max!=='')value=Math.min(Number(input.max),value);}
  if(el[key]===value)return;
  const before=snapshot();input._before=input._before||before;
  if(el.type==='vector'&&key==='strokeWidth'&&el.stroke==null)el.stroke=vectorValue(el,'stroke','#233c47');
  el[key]=value;
  if(['text','fontFamily','fontSize','fontWeight','fontStyle','color','align','lineHeight','shape','fill','stroke','strokeWidth'].includes(key)){if(el.appearanceSrc)el.originalAppearanceSrc=el.originalAppearanceSrc||el.appearanceSrc;delete el.appearanceSrc;}
  if(live&&input._grouped)state.history.pop();
  commit(before,{redraw:!live});
  if(live){state.history[state.history.length-1]=input._before;input._grouped=true;renderPage();renderLayers();}
}
function reorder(direction){const el=selected();if(!el||el.locked)return;mutate(()=>{const list=[...currentPage().elements].sort((a,b)=>a.zIndex-b.zIndex);const index=list.findIndex(e=>e.id===el.id),target=index+direction;if(target<0||target>=list.length)return;[list[index],list[target]]=[list[target],list[index]];list.forEach((entry,i)=>entry.zIndex=i);});}
function addElement(type,extras={}){if(state.compare)return;const page=currentPage();if(!page)return;const el={id:uid(),type,name:type==='text'?'新文字':type==='shape'?'新图形':'新图片',x:Math.max(20,page.width/2-75),y:Math.max(20,page.height/2-25),width:150,height:50,rotation:0,zIndex:Math.max(-1,...page.elements.map(e=>e.zIndex||0))+1,opacity:1,visible:true,locked:false,source:'manual',reviewRequired:false,...extras};mutate(()=>{page.elements.push(el);state.selected=el.id;});state.leftTab='layers';setLeftTab('layers');toast('已添加到页面，拖动可调整位置');}
function addText(){addElement('text',{text:'在这里输入文字',fontFamily:'Microsoft YaHei',fontSize:18,fontWeight:400,fontStyle:'normal',color:'#233c47',align:'left',lineHeight:1.4,width:180,height:36});}
function addShape(){addElement('shape',{shape:'rect',fill:'#e4eddb',stroke:'#7ca186',strokeWidth:1.5,width:120,height:80});}
async function imageData(file){if(!file)return null;if(!/^image\/(png|jpeg|webp|bmp)$/.test(file.type)){toast('请选择 PNG、JPEG、WebP 或 BMP 图片',true);return null;}if(file.size>20*1024*1024){toast('单张替换图片请小于 20 MB',true);return null;}return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error('图片读取失败'));reader.readAsDataURL(file);});}
async function addImage(file){try{const src=await imageData(file);if(!src)return;const img=new Image();img.src=src;await img.decode();const width=Math.min(240,currentPage().width*.5),height=width*img.height/img.width;addElement('image',{src,width,height,name:file.name});}catch(error){toast(error.message,true);}}
async function replaceImage(file,el){if(el.locked)return;try{const src=await imageData(file);if(!src)return;mutate(()=>{el.src=src;delete el.appearanceSrc;el.source='manual';});toast('图片已替换，布局保持不变');}catch(error){toast(error.message,true);}}
function deleteElement(){const el=selected();if(!el)return;if(el.locked){toast('请先解锁此元素');return;}mutate(()=>{currentPage().elements=currentPage().elements.filter(e=>e.id!==el.id);state.selected=null;});}
function duplicateElement(){const el=selected();if(!el)return;const copy=clone(el);copy.id=uid();copy.name=(copy.name||TYPE_NAMES[copy.type])+' 副本';copy.x+=12;copy.y+=12;copy.locked=false;copy.zIndex=Math.max(...currentPage().elements.map(e=>e.zIndex||0))+1;mutate(()=>{currentPage().elements.push(copy);state.selected=copy.id;});}
function undo(){if(!state.history.length||state.compare)return;state.future.push(snapshot());state.doc=JSON.parse(state.history.pop());state.page=Math.min(state.page,state.doc.pages.length-1);if(!selected())state.selected=null;state.revision++;refreshStats();scheduleSave();saveLabel('待保存…');renderEditor();}
function redo(){if(!state.future.length||state.compare)return;state.history.push(snapshot());state.doc=JSON.parse(state.future.pop());state.page=Math.min(state.page,state.doc.pages.length-1);if(!selected())state.selected=null;state.revision++;refreshStats();scheduleSave();saveLabel('待保存…');renderEditor();}
function setLeftTab(tab){state.leftTab=tab;$$('[data-left-tab]').forEach(button=>button.classList.toggle('active',button.dataset.leftTab===tab));$('#pages-list').classList.toggle('hidden',tab!=='pages');$('#layers-list').classList.toggle('hidden',tab!=='layers');}
function renderOutput(){const stats=docStats();$('#output-summary').innerHTML=`${icon('file')}<div class="output-doc"><strong>${escapeHTML(state.doc.name)}</strong><small>当前编辑版本 · 所有格式均从当前数据模型导出</small></div><div class="output-stat"><strong>${stats.pages}</strong><span>页面</span></div><div class="output-stat"><strong>${stats.elements}</strong><span>可编辑元素</span></div><div class="output-stat"><strong>${stats.review}</strong><span>建议复核</span></div>`;const warnings=[...(state.doc.warnings||[]),...state.doc.pages.flatMap(p=>p.warnings||[])];const unique=[...new Set(warnings)];$('#output-warnings').innerHTML=`<p>特殊图案与未识别内容保留原始图像外观；文字内容和样式修改后按新的字体渲染。${stats.review?`还有 ${stats.review} 个元素建议核对原稿。`:''}</p>${unique.slice(0,7).map(w=>`<div class="warning-note">${escapeHTML(w)}</div>`).join('')}`;}
async function exportDocument(format){
  if(!state.doc||state.busy)return;
  const docId=state.doc.id,extension=format==='pdf-text'?'pdf':format,label=format==='pdf-text'?'可编辑文本 PDF':format.toUpperCase();
  busy(true,`正在导出 ${label}`,'正在保存当前修改并生成可下载文件。');
  try{
    await flushSave();const rev=state.revision;
    const response=await api(`/api/documents/${encodeURIComponent(docId)}/export/${format}`,{method:'POST',headers:{'Content-Type':'application/json'},body:snapshot()});
    const blob=await operationProgress.download(response),url=URL.createObjectURL(blob);
    triggerDownload(url,(state.doc.name.replace(/\.[^.]+$/,'').replace(/[\\/:*?"<>|]/g,'_')||'document')+'.'+extension);
    setTimeout(()=>URL.revokeObjectURL(url),60000);
    if(state.revision!==rev)scheduleSave();toast(`${label} 已导出`);refreshDocuments();
  }catch(error){toast(error.message,true);}finally{busy(false);}
}
function openJSON(){refreshStats();$('#json-editor').value=JSON.stringify(state.doc,null,2);$('#json-error').textContent='';$('#json-dialog').showModal();}
function validateDocument(doc){if(!doc||typeof doc!=='object')throw new Error('根节点必须是文档对象');if(doc.id!==state.doc.id)throw new Error('不能修改文档 ID');if(doc.schemaVersion!==1)throw new Error('schemaVersion 必须为 1');if(!Array.isArray(doc.pages)||!doc.pages.length||doc.pages.length>100)throw new Error('pages 必须包含 1 至 100 页');const ids=new Set();doc.pages.forEach((page,i)=>{if(!Number.isFinite(page.width)||!Number.isFinite(page.height)||page.width<=0||page.height<=0)throw new Error(`第 ${i+1} 页尺寸无效`);if(!Array.isArray(page.elements))throw new Error(`第 ${i+1} 页缺少 elements 数组`);page.elements.forEach(el=>{if(!el.id||ids.has(el.id))throw new Error('元素 ID 为空或重复');ids.add(el.id);if(!TYPE_NAMES[el.type])throw new Error(`不支持的元素类型：${el.type}`);for(const key of ['x','y','width','height'])if(!Number.isFinite(el[key]))throw new Error(`元素 ${el.id} 的 ${key} 必须为有限数值`);if(el.width<=0||el.height<=0)throw new Error(`元素 ${el.id} 的尺寸必须大于 0`);});});}
async function applyJSON(){let doc;try{doc=JSON.parse($('#json-editor').value);validateDocument(doc);}catch(error){$('#json-error').textContent=error.message;return;}const button=$('#json-apply');button.disabled=true;try{await flushSave();const before=snapshot();const response=await api(`/api/documents/${encodeURIComponent(doc.id)}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(doc)});doc=await response.json();state.doc=doc;refreshStats();state.page=Math.min(state.page,doc.pages.length-1);state.selected=null;const changed=before!==snapshot();if(changed){state.history.push(before);state.future=[];state.revision++;state.savedRevision=state.revision;}saveLabel('已保存到本机');$('#json-dialog').close();renderEditor();if(state.screen==='output')renderOutput();toast('数据已验证，画布已同步更新');}catch(error){$('#json-error').textContent=error.message;}finally{button.disabled=false;}}
function bindEvents(){icons();$$('[data-screen]').forEach(button=>button.onclick=()=>showScreen(button.dataset.screen));$('.brand').onclick=event=>{event.preventDefault();showScreen('input');};$('#choose-files').onclick=event=>{event.stopPropagation();$('#file-input').click();};$('#dropzone').onclick=()=>$('#file-input').click();$('#dropzone').onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();$('#file-input').click();}};$('#file-input').onchange=event=>{setFiles(event.target.files);event.target.value='';};const dropzone=$('#dropzone');['dragenter','dragover'].forEach(type=>dropzone.addEventListener(type,event=>{event.preventDefault();dropzone.classList.add('dragover');}));['dragleave','drop'].forEach(type=>dropzone.addEventListener(type,event=>{event.preventDefault();dropzone.classList.remove('dragover');}));dropzone.addEventListener('drop',event=>setFiles(event.dataTransfer.files));$('#parse-files').onclick=parseFiles;$('#load-demo').onclick=loadDemo;$('#refresh-documents').onclick=refreshDocuments;$('#go-output').onclick=()=>showScreen('output');$('#back-editor').onclick=()=>showScreen('editor');$('#document-name').onchange=event=>mutate(()=>state.doc.name=event.target.value.trim()||'未命名文档');$('#undo').onclick=undo;$('#redo').onclick=redo;$('#add-text').onclick=addText;$('#add-shape').onclick=addShape;$('#add-image').onclick=()=>$('#image-input').click();$('#image-input').onchange=event=>{addImage(event.target.files[0]);event.target.value='';};$('#zoom-in').onclick=()=>setZoom(state.zoom+.1);$('#zoom-out').onclick=()=>setZoom(state.zoom-.1);$('#zoom-fit').onclick=fitZoom;$('#compare-toggle').onclick=()=>{if(state.editing)document.activeElement?.blur();state.compare=!state.compare;state.selected=null;renderEditor(false);};$$('[data-left-tab]').forEach(button=>button.onclick=()=>setLeftTab(button.dataset.leftTab));$('#page-canvas').onpointerdown=event=>{if(event.target===$('#page-canvas'))selectElement(null);};$('#canvas-wrap').onpointerdown=event=>{if(event.target===$('#canvas-wrap')||event.target===$('#canvas-stage'))selectElement(null);};$('#canvas-wrap').addEventListener('wheel',event=>{if(event.ctrlKey||event.metaKey){event.preventDefault();setZoom(state.zoom+(event.deltaY<0?.05:-.05));}},{passive:false});$('#json-open').onclick=openJSON;$('#json-close').onclick=()=>$('#json-dialog').close();$('#json-apply').onclick=applyJSON;$('#json-copy').onclick=async()=>{try{await navigator.clipboard.writeText($('#json-editor').value);toast('JSON 已复制');}catch{const editor=$('#json-editor');editor.select();toast('请按 Ctrl+C 复制选中的数据');}};$$('[data-export]').forEach(button=>button.onclick=()=>exportDocument(button.dataset.export));$('#save-state').onclick=()=>flushSave().then(()=>toast('已保存到本机')).catch(error=>toast(error.message,true));document.addEventListener('keydown',keyboard);window.addEventListener('beforeunload',event=>{if(state.doc&&state.savedRevision<state.revision){event.preventDefault();event.returnValue='';}});window.addEventListener('resize',()=>{if(state.screen==='editor'&&state.doc)renderPage();});}
function keyboard(event){if(state.screen!=='editor'||!state.doc||state.busy||$('#json-dialog').open)return;const tag=document.activeElement?.tagName;if(['INPUT','TEXTAREA','SELECT'].includes(tag)||document.activeElement?.isContentEditable)return;const cmd=event.ctrlKey||event.metaKey;if(cmd&&event.key.toLowerCase()==='s'){event.preventDefault();flushSave().then(()=>toast('已保存到本机')).catch(error=>toast(error.message,true));return;}if(state.compare)return;if(cmd&&event.key.toLowerCase()==='z'){event.preventDefault();event.shiftKey?redo():undo();return;}if(cmd&&event.key.toLowerCase()==='y'){event.preventDefault();redo();return;}if(event.key==='Delete'||event.key==='Backspace'){event.preventDefault();deleteElement();return;}if(event.key==='Escape'){selectElement(null);return;}if(cmd&&event.key.toLowerCase()==='d'){event.preventDefault();duplicateElement();return;}if(cmd&&event.key.toLowerCase()==='c'&&selected()){event.preventDefault();state.clipboard=clone(selected());toast('元素已复制');return;}if(cmd&&event.key.toLowerCase()==='v'&&state.clipboard){event.preventDefault();const copy=clone(state.clipboard);copy.id=uid();copy.x+=15;copy.y+=15;copy.locked=false;copy.zIndex=Math.max(-1,...currentPage().elements.map(e=>e.zIndex||0))+1;mutate(()=>{currentPage().elements.push(copy);state.selected=copy.id;});return;}const el=selected();if(el&&!el.locked&&['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){event.preventDefault();const step=event.shiftKey?10:1;mutate(()=>{if(event.key==='ArrowLeft')el.x-=step;if(event.key==='ArrowRight')el.x+=step;if(event.key==='ArrowUp')el.y-=step;if(event.key==='ArrowDown')el.y+=step;});}}
async function init(){
  bindEvents();
  $('#rebuild-format').addEventListener('change',updateImportControls);
  $$('[data-import-mode]').forEach(input=>input.addEventListener('change',()=>setImportMode(input.value)));
  $('#page-range').addEventListener('input',()=>{updateImportControls();showImportError(importValidationError());});
  $('#ocr-toggle').addEventListener('change',()=>{if(IMPORT_MODES[state.importMode].forceOCR)$('#ocr-toggle').checked=true;else state.optionalOCR=$('#ocr-toggle').checked;updateImportControls();});
  window.addEventListener('pagehide',event=>{if(!event.persisted)releaseReadingResult();});
  setImportMode('editor');refreshDocuments();loadFontFamilies();
  try{const health=await(await api('/api/health')).json();const ocr=health.ocr;const available=typeof ocr==='boolean'?ocr:ocr?.available??ocr?.enabled??false;$('#ocr-status').textContent=available?'OCR 引擎已就绪':'原生 PDF 可解析 · OCR 状态见结果';$('#ocr-status').title=typeof ocr==='object'?JSON.stringify(ocr):String(ocr);}catch{$('#ocr-status').textContent='等待本地解析服务';}
}
init();



