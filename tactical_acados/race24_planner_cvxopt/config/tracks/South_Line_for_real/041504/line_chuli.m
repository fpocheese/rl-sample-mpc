% 读取 CSV 文件
data = readmatrix('RightLine.csv'); % 请替换为您的实际文件路径

% 创建第一个图形窗口：以第1列为x轴，绘制第12列和第18列
figure(1);
set(gcf, 'Position', [100, 100, 800, 400]);

% subplot(1, 2, 1);
plot(data(:, 1), (mod(data(:, 12) + pi, 2*pi) - pi)*57.3, 'b-', 'LineWidth', 1.5);
xlabel('S');
ylabel('yaw');
title('yaw');
grid on;
set(gca, 'FontSize', 14);

figure(2);
set(gcf, 'Position', [100, 100, 800, 400]);

% subplot(1, 2, 2);
plot(data(:, 1), data(:, 18), 'r-', 'LineWidth', 1.5);
xlabel('S');
ylabel('speed(m/s)');
title('speed');
grid on;
set(gca, 'FontSize', 14);

% 创建第二个图形窗口：以第10列为x轴，第11列为y轴，并标注第1列数值
figure(3);
set(gcf, 'Position', [100, 100, 800, 600]);

% 绘制路径图
plot(data(:, 10), data(:, 11), 'k-o', 'LineWidth', 1, 'MarkerSize', 4, 'MarkerFaceColor', 'g');
xlabel('X)');
ylabel('Y');
title('位置对应的S距离');
grid on;
axis equal;
hold on; % 保持图形
set(gca, 'FontSize', 14);

% 添加第1列的数值标签
label_step = max(1, floor(size(data, 1)/80)); % 自动计算标签密度（最多显示20个标签）
for i = 1:label_step:size(data, 1)
    text(data(i, 10), data(i, 11), num2str(data(i, 1)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', ...
        'FontSize', 10, 'Color', 'blue');
end

% 可选：突出显示起点和终点
if size(data, 1) > 1
    % 标注起点
    text(data(1, 10), data(1, 11), ['Start: ' num2str(data(1, 1))], ...
        'VerticalAlignment', 'bottom', 'FontSize', 9, 'Color', 'red', 'FontWeight', 'bold');
    
    % 标注终点
    text(data(end, 10), data(end, 11), ['End: ' num2str(data(end, 1))], ...
        'VerticalAlignment', 'top', 'FontSize', 9, 'Color', 'red', 'FontWeight', 'bold');
end

hold off;




figure(4);
set(gcf, 'Position', [100, 100, 800, 600]);

% 绘制路径图
plot(data(:, 10), data(:, 11), 'k-o', 'LineWidth', 1, 'MarkerSize', 4, 'MarkerFaceColor', 'g');
xlabel('X');
ylabel('Y');
title('位置对应的速度');
grid on;
axis equal;
hold on; % 保持图形
set(gca, 'FontSize', 14);

% 添加第1列的数值标签
label_step = max(1, floor(size(data, 1)/80)); % 自动计算标签密度（最多显示20个标签）
for i = 1:label_step:size(data, 1)
    text(data(i, 10), data(i, 11), num2str(data(i, 18)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', ...
        'FontSize', 10, 'Color', 'blue');
end

% 可选：突出显示起点和终点
if size(data, 18) > 1
    % 标注起点
    text(data(1, 10), data(1, 11), ['Start: ' num2str(data(1, 1))], ...
        'VerticalAlignment', 'bottom', 'FontSize', 9, 'Color', 'red', 'FontWeight', 'bold');
    
    % 标注终点
    text(data(end, 10), data(end, 11), ['End: ' num2str(data(end, 1))], ...
        'VerticalAlignment', 'top', 'FontSize', 9, 'Color', 'red', 'FontWeight', 'bold');
end

hold off;