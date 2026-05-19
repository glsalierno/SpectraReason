function make_figures()
% Auto-generated FTIR report figures (visualization only; assignments unchanged).
% Requires CSV exports in the same folder as this file.
% Run: cd('.../matlab_export'); make_figures

showPeakLabels = true;
labelAllLabeledPeaks = true;  % all is_labeled==1 rows in peaks CSV (matches HTML report)
maxPeakLabels = 48;           % cap when labelAllLabeledPeaks is false
showSeparatePanels = true;    % region guide + spectrum peaks as separate PNGs (recommended)
showStackedFigure = false;    % optional legacy stacked spectrum+Kronecker
showRulerOverlay = false;     % ruler bands on spectrum (off when using separate region guide)
showKronecker = false;
reverseX = true;
outputFormat = 'png';
exportDpi = 300;
closeFiguresAfterExport = false;  % false = keep figures open for resize/review in MATLAB
bringFiguresToFront = true;       % focus each new figure when kept open

% --- User-tuned typography (edit here; preserved across report regen) ---
fontRegionBand = 8;
fontRegionAxis = 10;
fontSpectrumPeakLabel = 14;
fontSpectrumAxis = 18;
fontCombinedPeakLabel = 8;
fontCombinedAxis = 10;
fontCombinedRulerOverlay = 18;
peakLabelRotation = -90;          % vertical wavenumber labels (degrees)
peakLabelCollisionAware = true;   % stagger vertically when labels overlap

dataDir = char(fileparts(mfilename('fullpath')));
outDir = char(fullfile(dataDir, '..', 'presentation', 'figures'));
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

spectrumList = {'Catechol-120-80-9-IR', 'Nylon_T', 'Benzoic_acid_-_65-85-0-IR', '1H-Pyrrole-2-carboxylic_acid-634-97-9-IR', '1H-Indol-5-ol-1953-54-4-IR', 'Indole_120-72-9-IR', 'Polydopamine_Powder', 'Dopamine_Powder'};

for si = 1:numel(spectrumList)
    stem = normalizeStem(spectrumList{si});
    if showSeparatePanels
        plotRegionGuide(stem, dataDir, outDir);
        plotSpectrumPeaks(stem, dataDir, outDir);
    end
    if showStackedFigure
        plotOneSpectrum(stem, dataDir, outDir);
    end
end

function stemChar = normalizeStem(stem)
    if iscell(stem)
        stem = stem{1};
    end
    stemChar = char(strtrim(string(stem)));
    if isempty(stemChar)
        stemChar = 'spectrum';
    end
end

function csvPath = csvPathFor(dataDir, stem, suffix)
    csvPath = char(fullfile(dataDir, [stem suffix]));
end

function warnMissing(msg, pathChar)
    warning('FTIR:make_figures:MissingFile', '%s: %s', msg, pathChar);
end

function setWavenumberXLim(ax, wn)
    % xlim requires increasing limits; use XDir reverse for IR high-to-low display.
    xlim(ax, [min(wn) max(wn)]);
    if reverseX
        set(ax, 'XDir', 'reverse');
    end
end

function exportFigure(figOrLayout, outFile)
    outFile = char(outFile);
    outFolder = fileparts(outFile);
    if ~isempty(outFolder) && ~exist(outFolder, 'dir')
        mkdir(outFolder);
    end
    figHandle = ancestor(figOrLayout, 'figure');
    if isempty(figHandle) || ~ishandle(figHandle)
        figHandle = gcf;
    end
    try
        exportgraphics(figOrLayout, outFile, 'Resolution', exportDpi);
    catch
        print(figOrLayout, outFile, ['-d' char(outputFormat)], sprintf('-r%d', exportDpi));
    end
    fprintf('Wrote %s\n', outFile);
    if ~closeFiguresAfterExport
        if bringFiguresToFront
            figure(figHandle);
        end
    else
        close(figHandle);
    end
end

function plotRegionGuide(stem, dataDir, outDir)
    rulerFile = csvPathFor(dataDir, stem, '_ruler_regions.csv');
    specFile = csvPathFor(dataDir, stem, '_spectrum.csv');
    if ~isfile(rulerFile)
        warnMissing('Missing ruler CSV', rulerFile);
        return;
    end
    if ~isfile(specFile)
        warnMissing('Missing spectrum CSV (for wavenumber axis)', specFile);
        return;
    end
    T = readtable(specFile);
    wn = T.wavenumber_cm1;
    R = readtable(rulerFile);
    nR = height(R);
    figH = max(3.5, 1.2 + 0.28 * nR);
    fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 16 figH]);
    ax = axes(fig);
    hold(ax, 'on');
    yLo = 0.02;
    yHi = 0.98;
    rowH = (yHi - yLo) / max(nR, 1);
    for ri = 1:nR
        y0 = yHi - ri * rowH + 0.12 * rowH;
        y1 = yHi - (ri - 1) * rowH - 0.12 * rowH;
        lo = R.lo_cm1(ri);
        hi = R.hi_cm1(ri);
        fill(ax, [lo hi hi lo], [y0 y0 y1 y1], [0.89 0.91 0.94], ...
            'EdgeColor', [0.58 0.64 0.72], 'LineWidth', 0.6);
        lbl = char(string(R.region_label(ri)));
        text(ax, (lo + hi) / 2, (y0 + y1) / 2, lbl, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontSize', fontRegionBand, 'Color', [0.2 0.25 0.33], 'Interpreter', 'none');
    end
    hold(ax, 'off');
    ylim(ax, [0 1]);
    set(ax, 'YTick', []);
    xlabel(ax, 'Wavenumber (cm^{-1})');
    title(ax, 'FTIR region guide (tentative ranges)', 'FontWeight', 'normal');
    set(ax, 'Box', 'on', 'FontName', 'Arial', 'FontSize', fontRegionAxis);
    setWavenumberXLim(ax, wn);
    outFile = fullfile(outDir, [stem '_region_guide_matlab.' char(outputFormat)]);
    exportFigure(fig, outFile);
end

function drawSpectrumPeakLabels(ax, wnPk, yPk, txtLabels, ySpan, labelFontSize)
    if nargin < 6 || isempty(labelFontSize)
        labelFontSize = fontSpectrumPeakLabel;
    end
    if isempty(wnPk)
        return;
    end
    laid = layoutPeakLabels(wnPk, yPk, txtLabels, ySpan);
    for li = 1:numel(laid)
        text(ax, laid(li).x, laid(li).yText, laid(li).text, ...
            'HorizontalAlignment', laid(li).hAlign, ...
            'VerticalAlignment', laid(li).vAlign, ...
            'FontSize', labelFontSize, ...
            'Color', [0.2 0.2 0.2], ...
            'Rotation', laid(li).angle);
    end
end

function laid = layoutPeakLabels(wnPk, yPk, txtLabels, ySpan)
    n = numel(wnPk);
    laid = repmat(struct('x', 0, 'yText', 0, 'text', '', 'hAlign', 'center', ...
        'vAlign', 'bottom', 'angle', peakLabelRotation), n, 1);
    yMax = max(yPk);
    yMin = min(yPk);
    ySpan = max(max(ySpan, yMax - yMin), 1e-9);
    wnSpan = max(max(wnPk) - min(wnPk), 400);
    yCeil = yMax + 0.18 * ySpan;
    baseShift = 10;
    for i = 1:n
        laid(i).x = wnPk(i);
        laid(i).text = txtLabels{i};
        laid(i).yText = yPk(i) + (baseShift / 280) * ySpan;
        laid(i).angle = peakLabelRotation;
    end
    if ~peakLabelCollisionAware || n <= 1
        return;
    end
    [~, ord] = sort(yPk, 'descend');
    shiftsPx = [10 20 32 44 58 72 88 104];
    placed = zeros(0, 4);
    for oi = 1:numel(ord)
        i = ord(oi);
        wn = wnPk(i);
        y = yPk(i);
        txt = txtLabels{i};
        ok = false;
        for si = 1:numel(shiftsPx)
            ysh = shiftsPx(si);
            box = estimateLabelBox(wn, y, txt, ySpan, wnSpan, peakLabelRotation, ysh, yCeil);
            if isempty(box) || anyBoxOverlaps(box, placed)
                continue;
            end
            placed(end+1, :) = box; %#ok<AGROW>
            laid(i).angle = peakLabelRotation;
            laid(i).yText = y + (ysh / 280) * ySpan;
            laid(i).hAlign = 'center';
            laid(i).vAlign = 'bottom';
            ok = true;
            break;
        end
    end
end

function box = estimateLabelBox(wn, y, txt, ySpan, wnSpan, angle, yshiftPx, yCeil)
    yOff = (yshiftPx / 280) * ySpan;
    if abs(angle) >= 89
        xHalf = max(5, 7) * (wnSpan / max(ySpan * 50, 1)) * 0.06;
        charH = 0.026 * ySpan * max(numel(txt), 3);
        yTop = y + yOff + charH + 0.02 * ySpan;
        yBot = y + yOff - 0.01 * ySpan;
        if yTop > yCeil
            box = [];
            return;
        end
        box = [wn - xHalf, wn + xHalf, yBot, yTop];
        return;
    end
    wHalf = max(12, 10 + numel(txt) * 1.6);
    if abs(angle) >= 45
        wHalf = max(10, 8 + numel(txt) * 0.9);
    end
    xHalf = wHalf * (wnSpan / max(ySpan * 50, 1)) * 0.15;
    yTop = y + yOff + 0.045 * ySpan;
    yBot = y + yOff - 0.012 * ySpan;
    if abs(angle) < 89 && yTop > yCeil
        box = [];
        return;
    end
    box = [wn - xHalf, wn + xHalf, yBot, yTop];
end

function tf = anyBoxOverlaps(box, placed)
    tf = false;
    if isempty(placed)
        return;
    end
    for r = 1:size(placed, 1)
        b = placed(r, :);
        if box(1) < b(2) && b(1) < box(2) && box(3) < b(4) && b(3) < box(4)
            tf = true;
            return;
        end
    end
end

function plotSpectrumPeaks(stem, dataDir, outDir)
    specFile = csvPathFor(dataDir, stem, '_spectrum.csv');
    peaksFile = csvPathFor(dataDir, stem, '_peaks.csv');
    if ~isfile(specFile)
        warnMissing('Missing spectrum CSV', specFile);
        return;
    end
    T = readtable(specFile);
    wn = T.wavenumber_cm1;
    y = T.absorbance;
    nLabels = 0;
    if showPeakLabels && isfile(peaksFile)
        P = readtable(peaksFile);
        if ismember('is_labeled', P.Properties.VariableNames)
            idx = find(P.is_labeled == 1);
        else
            idx = (1:min(height(P), maxPeakLabels))';
        end
        if ~labelAllLabeledPeaks
            idx = idx(1:min(numel(idx), maxPeakLabels));
        end
        nLabels = numel(idx);
    end
    figH = 12 + min(6, max(0, nLabels - 18) * 0.12);
    fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 16 figH]);
    ax = axes(fig);
    hold(ax, 'on');
    plot(ax, wn, y, 'Color', [0 0.447 0.741], 'LineWidth', 1.1);
    if showPeakLabels && isfile(peaksFile)
        P = readtable(peaksFile);
        if ismember('is_labeled', P.Properties.VariableNames)
            idx = find(P.is_labeled == 1);
        else
            idx = (1:min(height(P), maxPeakLabels))';
        end
        if ~labelAllLabeledPeaks
            idx = idx(1:min(numel(idx), maxPeakLabels));
        end
        ySpan = max(y) - min(y);
        wnPk = P.peak_position_cm1(idx);
        yPk = P.peak_height(idx);
        txtLabels = arrayfun(@(x) sprintf('%.0f', x), wnPk, 'UniformOutput', false);
        plot(ax, wnPk, yPk, 'o', 'MarkerSize', 4, 'Color', [0.85 0.33 0.1]);
        drawSpectrumPeakLabels(ax, wnPk, yPk, txtLabels, ySpan);
    end
    hold(ax, 'off');
    xlabel(ax, 'Wavenumber (cm^{-1})');
    ylabel(ax, 'Normalized absorbance');
    title(ax, strrep(stem, '_', ' '), 'FontWeight', 'normal');
    set(ax, 'Box', 'on', 'FontName', 'Arial', 'FontSize', fontSpectrumAxis);
    grid(ax, 'on');
    setWavenumberXLim(ax, wn);
    outFile = fullfile(outDir, [stem '_spectrum_peaks_matlab.' char(outputFormat)]);
    exportFigure(fig, outFile);
end

function plotOneSpectrum(stem, dataDir, outDir)
    specFile = csvPathFor(dataDir, stem, '_spectrum.csv');
    peaksFile = csvPathFor(dataDir, stem, '_peaks.csv');
    rulerFile = csvPathFor(dataDir, stem, '_ruler_regions.csv');
    if ~isfile(specFile)
        warnMissing('Missing spectrum CSV', specFile);
        return;
    end
    T = readtable(specFile);
    wn = T.wavenumber_cm1;
    y = T.absorbance;
    if showKronecker
        tl = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    else
        tl = tiledlayout(1, 1, 'Padding', 'compact');
    end
    ax1 = nexttile(tl, 1);
    hold(ax1, 'on');
    plot(ax1, wn, y, 'Color', [0 0.447 0.741], 'LineWidth', 1.2);
    if showRulerOverlay && isfile(rulerFile)
        R = readtable(rulerFile);
        yR = max(y) * 1.02;
        for ri = 1:height(R)
            lo = R.lo_cm1(ri);
            hi = R.hi_cm1(ri);
            patch(ax1, [lo hi hi lo], [yR*0.98 yR*0.98 yR yR], ...
                [0.85 0.85 0.85], 'EdgeColor', [0.4 0.4 0.4], 'FaceAlpha', 0.35);
            text(ax1, (lo+hi)/2, yR*1.01, char(string(R.region_label(ri))), ...
                'HorizontalAlignment', 'center', 'FontSize', fontCombinedRulerOverlay, 'Color', [0.2 0.2 0.2]);
        end
    end
    if showPeakLabels && isfile(peaksFile)
        P = readtable(peaksFile);
        if ismember('is_labeled', P.Properties.VariableNames)
            idx = find(P.is_labeled == 1);
        else
            idx = (1:min(height(P), maxPeakLabels))';
        end
        if ~labelAllLabeledPeaks
            idx = idx(1:min(numel(idx), maxPeakLabels));
        end
        ySpan = max(y) - min(y);
        wnPk = P.peak_position_cm1(idx);
        yPk = P.peak_height(idx);
        txtLabels = arrayfun(@(x) sprintf('%.0f', x), wnPk, 'UniformOutput', false);
        plot(ax1, wnPk, yPk, 'o', 'MarkerSize', 5, 'Color', [0.85 0.33 0.1]);
        drawSpectrumPeakLabels(ax1, wnPk, yPk, txtLabels, ySpan, fontCombinedPeakLabel);
    end
    hold(ax1, 'off');
    xlabel(ax1, 'Wavenumber (cm^{-1})');
    ylabel(ax1, 'Normalized absorbance');
    title(ax1, strrep(stem, '_', ' '), 'FontWeight', 'normal');
    set(ax1, 'Box', 'on', 'FontName', 'Arial', 'FontSize', fontCombinedAxis);
    grid(ax1, 'on');
    setWavenumberXLim(ax1, wn);
    if showKronecker && isfile(peaksFile)
        ax2 = nexttile(tl, 2);
        P = readtable(peaksFile);
        stem(ax2, P.peak_position_cm1, P.peak_height, 'Color', [0.3 0.5 0.7], 'LineWidth', 0.8);
        xlabel(ax2, 'Wavenumber (cm^{-1})');
        ylabel(ax2, 'Peak height');
        set(ax2, 'Box', 'on', 'FontName', 'Arial', 'FontSize', 10);
        grid(ax2, 'on');
        setWavenumberXLim(ax2, wn);
    end
    outFile = fullfile(outDir, [stem '_combined_matlab.' char(outputFormat)]);
    exportFigure(tl, outFile);
end

end
