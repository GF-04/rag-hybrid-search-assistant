// assets/charts.js — 对抗性对照实验数据
(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  // --- Chart 1: 两种模式 Top1 准确率对比（按问题类型） ---
  var chart1 = echarts.init(document.getElementById('chart-accuracy'), null, { renderer: 'svg' });
  chart1.setOption({
    animation: false,
    tooltip: { trigger: 'axis', appendToBody: true, axisPointer: { type: 'shadow' } },
    legend: { data: ['模式A 纯向量', '模式B 混合检索'], textStyle: { color: ink }, top: 0 },
    grid: { left: '3%', right: '4%', bottom: '3%', top: 40, containLabel: true },
    xAxis: {
      type: 'category',
      data: ['普通语义', '同义改写', '精确编号', '对抗性混合'],
      axisLabel: { color: ink, fontSize: 13 },
      axisLine: { lineStyle: { color: rule } }
    },
    yAxis: {
      type: 'value',
      max: 100,
      name: 'Top1 准确率 (%)',
      nameTextStyle: { color: muted },
      axisLabel: { color: muted, formatter: '{value}%' },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [
      {
        name: '模式A 纯向量',
        type: 'bar',
        data: [100, 75, 100, 33],
        itemStyle: { color: accent, borderRadius: [6, 6, 0, 0] },
        label: { show: true, position: 'top', color: ink, formatter: '{c}%' },
        barWidth: 30
      },
      {
        name: '模式B 混合检索',
        type: 'bar',
        data: [100, 50, 100, 33],
        itemStyle: { color: accent2, borderRadius: [6, 6, 0, 0] },
        label: { show: true, position: 'top', color: ink, formatter: '{c}%' },
        barWidth: 30
      }
    ]
  });
  window.addEventListener('resize', function() { chart1.resize(); });

  // --- Chart 2: 精确编号问题三路召回名次对比 ---
  var chart2 = echarts.init(document.getElementById('chart-rank'), null, { renderer: 'svg' });
  chart2.setOption({
    animation: false,
    tooltip: { trigger: 'axis', appendToBody: true, axisPointer: { type: 'shadow' } },
    legend: { data: ['向量召回名次', 'BM25召回名次', 'RRF融合后名次'], textStyle: { color: ink }, top: 0 },
    grid: { left: '3%', right: '4%', bottom: '3%', top: 40, containLabel: true },
    xAxis: {
      type: 'category',
      data: ['ERR_404_A 是什么错误', 'HNSW-M16 efSearch', 'V2.3.1-hotfix 修复'],
      axisLabel: { color: ink, fontSize: 12 },
      axisLine: { lineStyle: { color: rule } }
    },
    yAxis: {
      type: 'value',
      inverse: true,
      name: '召回名次（越小越好）',
      nameTextStyle: { color: muted },
      min: 1,
      max: 6,
      axisLabel: { color: muted },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [
      {
        name: '向量召回名次',
        type: 'bar',
        data: [1, 1, 1],
        itemStyle: { color: accent, borderRadius: [4, 4, 0, 0] },
        label: { show: true, position: 'inside', color: '#fff', fontWeight: 'bold' },
        barWidth: 24
      },
      {
        name: 'BM25召回名次',
        type: 'bar',
        data: [3, 1, 1],
        itemStyle: { color: accent2, borderRadius: [4, 4, 0, 0] },
        label: { show: true, position: 'inside', color: '#fff', fontWeight: 'bold' },
        barWidth: 24
      },
      {
        name: 'RRF融合后名次',
        type: 'bar',
        data: [1, 1, 1],
        itemStyle: { color: '#f59e0b', borderRadius: [4, 4, 0, 0] },
        label: { show: true, position: 'inside', color: '#fff', fontWeight: 'bold' },
        barWidth: 24
      }
    ]
  });
  window.addEventListener('resize', function() { chart2.resize(); });

  // --- Chart 3: 检索模式雷达图 ---
  var chart3 = echarts.init(document.getElementById('chart-radar'), null, { renderer: 'svg' });
  chart3.setOption({
    animation: false,
    tooltip: { appendToBody: true },
    legend: { data: ['模式A 纯向量', '模式B 混合检索'], textStyle: { color: ink }, top: 0 },
    radar: {
      indicator: [
        { name: '语义召回', max: 100 },
        { name: '精确匹配', max: 100 },
        { name: '抗语义干扰', max: 100 },
        { name: '召回多样性', max: 100 },
        { name: '工程可解释性', max: 100 }
      ],
      axisName: { color: ink, fontSize: 12 },
      splitLine: { lineStyle: { color: rule } },
      splitArea: { areaStyle: { color: [bg2, 'transparent'] } },
      axisLine: { lineStyle: { color: rule } }
    },
    series: [{
      type: 'radar',
      data: [
        {
          value: [95, 80, 55, 85, 90],
          name: '模式A 纯向量',
          areaStyle: { color: accent + '33' },
          lineStyle: { color: accent, width: 2 },
          itemStyle: { color: accent }
        },
        {
          value: [95, 92, 62, 90, 85],
          name: '模式B 混合检索',
          areaStyle: { color: accent2 + '33' },
          lineStyle: { color: accent2, width: 2 },
          itemStyle: { color: accent2 }
        }
      ]
    }]
  });
  window.addEventListener('resize', function() { chart3.resize(); });
})();
