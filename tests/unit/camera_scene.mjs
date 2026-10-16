// Run with `node tests/unit/camera_scene.mjs`; renderer boundary fixture, no WebGPU required.
import assert from 'node:assert/strict';
import fs from 'node:fs';
const source=fs.readFileSync(new URL('../../wrs_agent/nodes/camera/scene.js',import.meta.url),'utf8');
const {CameraScene}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
let live=0, allocated=0;
const buffer=()=>{live++;allocated++;return {destroyed:false,destroy(){
  assert.equal(this.destroyed,false,'no double destroy');this.destroyed=true;live--;
}};};
const renderer={models:new Map([['robot',{buffers:[]}]]),geometries:new Map([[0,{buffers:[]}]]),
  addGeometry(g){this.geometries.set(g.id,{...g,buffers:[buffer()]});},
  add(m){this.models.set(m.id,{...m,buffers:[buffer()]});},
  remove(id){this.models.get(id)?.buffers.forEach(b=>b.destroy());this.models.delete(id);},
  setTransform(id,tf){this.models.get(id).tf=tf;},
};
const camera={setTo(pos,target){this.pos=pos;this.target=target;}};
const scene=new CameraScene(()=>({renderer,camera}));
const value={observation_id:'obs.1',points:[[0,0,.25],[.1,.1,.3]],
  colors:[[235,70,210],[100,150,200]],segmented_point_count:1,
  candidate_set_id:'grasp.1',grasps:[{translation:[0,0,.25],rotation:[1,0,0,0,1,0,0,0,1],
  width_m:.04,height_m:.02,depth_m:.03}]};
const buffers=()=>[...renderer.models.values(),...renderer.geometries.values()].flatMap(m=>m.buffers);
scene.update(value);
assert.equal(renderer.geometries.size,3);assert.equal(renderer.models.size,9);assert.equal(live,10);
const colors=renderer.geometries.get('camera-overlay:cloud').colors;
value.colors.flat().forEach((c,i)=>assert.ok(Math.abs(colors[i]-c/255)<1e-6));
const cloud=renderer.models.get('camera-overlay:cloud').tf;
assert.deepEqual([...cloud.slice(0,3)],[0,-1,0]);
const left=renderer.models.get('camera-overlay:grasp:0:0').tf;
assert.ok(Math.abs(left[12]-.25)<1e-6 && Math.abs(left[13]-.695)<1e-6);
assert.ok(Math.abs(left[14]-.372)<1e-6,'jaw and cloud use the same display transform');
const before=allocated;
scene.update(value);assert.equal(allocated,before,'same candidate does not allocate again');
for(let i=0;i<200;i++) {
  const previous=buffers();
  scene.update({...value,candidate_set_id:`replacement.${i}`});
  assert.ok(previous.every(b=>b.destroyed),'replacement releases all previous GPU resources');
  assert.equal(live,10,'cloud and grasp buffers remain bounded');
}
scene.focus();assert.ok(camera.pos.every(Number.isFinite));
scene.clear();scene.clear();assert.equal(live,0);
assert.deepEqual([...renderer.models.keys()],['robot']);
assert.deepEqual([...renderer.geometries.keys()],[0]);
scene.update(value);
// WRS scene_init clears its renderer. The next observation must restore the overlay.
for(const m of renderer.models.values())m.buffers.forEach(b=>b.destroy());renderer.models.clear();
for(const g of renderer.geometries.values())g.buffers.forEach(b=>b.destroy());renderer.geometries.clear();
scene.update(value);assert.equal(live,10);scene.clear();assert.equal(live,0);
console.log('PASS: colors/transforms; 200 replacements destroy old buffers; clear and scene_init recovery');
