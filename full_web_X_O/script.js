const cells = Array.from(document.querySelectorAll('.cell'));
const btnNew = document.getElementById('btnNew');
const statusEl = document.getElementById('status');
const playX = document.getElementById('playX');
const playO = document.getElementById('playO');

function setStatus(t){ statusEl.textContent = t; }

async function api(path, opts={}){
  const res = await fetch(path, opts);
  return res.json();
}

function render(state){
  // board
  cells.forEach((el,i)=>{ el.textContent = state.board[i] === ' ' ? '' : state.board[i]; });
  // status
  if(state.game_over){
    if(state.winner === null){
      setStatus('انتهت بالتعادل');
    }else if(state.winner === state.human){
      setStatus('أحسنت! فزت ✅');
    }else{
      setStatus('الذكاء فاز 🤖');
    }
  }else{
    const turn = state.current === state.human ? 'دورك' : 'دور الذكاء';
    setStatus(`${turn} — أنت ${state.human}`);
  }
}

async function refresh(){
  const s = await api('/state');
  render(s);
  // toggle inputs according to state.human
  if(s.human === 'X'){ playX.checked = true; } else { playO.checked = true; }
}

async function reset(){
  const human = playO.checked ? 'O' : 'X';
  const s = await api('/reset', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ human })
  });
  render(s);
}

cells.forEach(el=>{
  el.addEventListener('click', async ()=>{
    const i = Number(el.dataset.i);
    const s0 = await api('/state');
    if(s0.game_over || s0.current !== s0.human || s0.board[i] !== ' ') return;
    const s = await api('/move', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ pos: i })
    });
    render(s);
  });
});

btnNew.addEventListener('click', reset);
playX.addEventListener('change', reset);
playO.addEventListener('change', reset);

refresh();
