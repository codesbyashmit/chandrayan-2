class MatchVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.sourceImg = null;
        this.referenceImg = null;
        this.matchData = null;
        this.scale = 1;
        this.showInliers = true;
        this.showOutliers = false;
    }


    async loadImages(sourceDataUrl, referenceDataUrl) {
        this.sourceImg = await this._loadImage(sourceDataUrl);
        this.referenceImg = await this._loadImage(referenceDataUrl);
    }


    setMatchData(data) {
        this.matchData = data;
    }


    draw() {
        if (!this.sourceImg || !this.referenceImg || !this.matchData) return;

        const srcW = this.sourceImg.width;
        const srcH = this.sourceImg.height;
        const refW = this.referenceImg.width;
        const refH = this.referenceImg.height;
        const maxH = Math.max(srcH, refH);
        const totalW = srcW + refW;


        const container = this.canvas.parentElement;
        const containerW = container.clientWidth - 32;
        this.scale = Math.min(1, containerW / totalW);

        const canvasW = Math.floor(totalW * this.scale);
        const canvasH = Math.floor(maxH * this.scale);

        this.canvas.width = canvasW;
        this.canvas.height = canvasH;
        this.canvas.style.width = canvasW + 'px';
        this.canvas.style.height = canvasH + 'px';


        this.ctx.fillStyle = '#050810';
        this.ctx.fillRect(0, 0, canvasW, canvasH);


        this.ctx.drawImage(this.sourceImg, 0, 0,
            Math.floor(srcW * this.scale),
            Math.floor(srcH * this.scale));
        this.ctx.drawImage(this.referenceImg,
            Math.floor(srcW * this.scale), 0,
            Math.floor(refW * this.scale),
            Math.floor(refH * this.scale));


        this.ctx.strokeStyle = 'rgba(96, 165, 250, 0.5)';
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([4, 4]);
        this.ctx.beginPath();
        this.ctx.moveTo(Math.floor(srcW * this.scale), 0);
        this.ctx.lineTo(Math.floor(srcW * this.scale), canvasH);
        this.ctx.stroke();
        this.ctx.setLineDash([]);


        this.ctx.font = '12px Outfit, sans-serif';
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
        this.ctx.fillText('SOURCE', 10, 20);
        this.ctx.fillText('REFERENCE', Math.floor(srcW * this.scale) + 10, 20);


        const allMatches = this.matchData.all_matches || {};
        const srcPts = allMatches.source_pts || [];
        const refPts = allMatches.reference_pts || [];
        const inlierMask = allMatches.inlier_mask || [];

        const offset = srcW * this.scale;


        if (this.showOutliers) {
            this.ctx.globalAlpha = 0.3;
            for (let i = 0; i < srcPts.length; i++) {
                if (!inlierMask[i]) {
                    this._drawMatch(
                        srcPts[i][0] * this.scale,
                        srcPts[i][1] * this.scale,
                        refPts[i][0] * this.scale + offset,
                        refPts[i][1] * this.scale,
                        'rgba(251, 113, 133, 0.6)',
                        1
                    );
                }
            }
            this.ctx.globalAlpha = 1.0;
        }


        if (this.showInliers) {
            for (let i = 0; i < srcPts.length; i++) {
                if (inlierMask[i]) {
                    this._drawMatch(
                        srcPts[i][0] * this.scale,
                        srcPts[i][1] * this.scale,
                        refPts[i][0] * this.scale + offset,
                        refPts[i][1] * this.scale,
                        'rgba(52, 211, 153, 0.7)',
                        1
                    );
                }
            }
        }


        const finalMatches = this.matchData.final_matches || {};
        const finalSrc = finalMatches.source_pts || [];
        const finalRef = finalMatches.reference_pts || [];

        for (let i = 0; i < finalSrc.length; i++) {
            this._drawMatch(
                finalSrc[i][0] * this.scale,
                finalSrc[i][1] * this.scale,
                finalRef[i][0] * this.scale + offset,
                finalRef[i][1] * this.scale,
                'rgba(96, 165, 250, 0.9)',
                2
            );
        }


        const inlierCount = inlierMask.filter(x => x).length;
        const totalCount = srcPts.length;
        this.ctx.font = 'bold 11px JetBrains Mono, monospace';
        this.ctx.fillStyle = 'rgba(96, 165, 250, 0.8)';
        this.ctx.fillText(
            `Final: ${finalSrc.length} | Inliers: ${inlierCount} | Total: ${totalCount}`,
            10, canvasH - 10
        );
    }

    _drawMatch(sx, sy, rx, ry, color, lineWidth) {

        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = lineWidth;
        this.ctx.beginPath();
        this.ctx.moveTo(sx, sy);
        this.ctx.lineTo(rx, ry);
        this.ctx.stroke();


        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.arc(sx, sy, 3, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.beginPath();
        this.ctx.arc(rx, ry, 3, 0, Math.PI * 2);
        this.ctx.fill();
    }

    _loadImage(dataUrl) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = dataUrl;
        });
    }
}



class DistributionVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
    }


    draw(gridCounts, stats) {
        if (!gridCounts || gridCounts.length === 0) return;

        const gridSize = gridCounts.length;
        const cellSize = 60;
        const padding = 60;

        const canvasW = gridSize * cellSize + padding * 2;
        const canvasH = gridSize * cellSize + padding * 2 + 80;

        this.canvas.width = canvasW;
        this.canvas.height = canvasH;


        this.ctx.fillStyle = '#0a0e1a';
        this.ctx.fillRect(0, 0, canvasW, canvasH);


        let maxCount = 0;
        for (let r = 0; r < gridSize; r++) {
            for (let c = 0; c < gridSize; c++) {
                maxCount = Math.max(maxCount, gridCounts[r][c]);
            }
        }
        maxCount = Math.max(maxCount, 1);


        for (let r = 0; r < gridSize; r++) {
            for (let c = 0; c < gridSize; c++) {
                const x = padding + c * cellSize;
                const y = padding + r * cellSize;
                const count = gridCounts[r][c];
                const intensity = count / maxCount;


                const hue = 120 + (1 - intensity) * 80;
                const sat = 60 + intensity * 30;
                const light = 10 + intensity * 40;

                this.ctx.fillStyle = `hsl(${hue}, ${sat}%, ${light}%)`;
                this.ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);


                this.ctx.strokeStyle = 'rgba(148, 163, 184, 0.15)';
                this.ctx.lineWidth = 1;
                this.ctx.strokeRect(x, y, cellSize, cellSize);


                if (count > 0) {
                    this.ctx.font = 'bold 14px JetBrains Mono, monospace';
                    this.ctx.fillStyle = intensity > 0.5 ? '#fff' : 'rgba(255,255,255,0.6)';
                    this.ctx.textAlign = 'center';
                    this.ctx.textBaseline = 'middle';
                    this.ctx.fillText(count, x + cellSize / 2, y + cellSize / 2);
                }
            }
        }


        this.ctx.font = 'bold 14px Outfit, sans-serif';
        this.ctx.fillStyle = '#f1f5f9';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Spatial Distribution of Match Points', canvasW / 2, 30);


        const statsY = padding + gridSize * cellSize + 30;
        this.ctx.font = '12px JetBrains Mono, monospace';
        this.ctx.textAlign = 'center';
        this.ctx.fillStyle = '#94a3b8';

        
        const statsText = [
            `Coverage: ${(stats.coverage * 100).toFixed(1)}%`,
            `Uniformity: ${(stats.uniformity_score * 100).toFixed(1)}%`,
            `Occupied: ${stats.occupied_cells}/${stats.total_cells} cells`,
            `Mean/cell: ${stats.mean_per_cell.toFixed(1)}`
        ].join('  |  ');

        this.ctx.fillText(statsText, canvasW / 2, statsY);
    }
}