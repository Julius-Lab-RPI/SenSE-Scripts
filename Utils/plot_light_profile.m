function plot_light_profile(t, I)
    plot(t, I)%, "LineWidth", 2);
    xticks(0:24:t(end))
    grid on
    xlabel("Time (h)")
end