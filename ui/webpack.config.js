const path = require("path")

module.exports  = {
    entry: ["./src/gridCounts.js"],
    resolve: {
        extensions: ['.js']
    },
    module: {
        rules: [
        {
            test: /\.css$/i,
            use: ['style-loader', 'css-loader'],
        },
        // ... other rules
        ]
    },
    devtool: 'inline-source-map',
    mode: 'development',
    output: {
        filename: 'main.js',
        path: path.resolve(__dirname, "dist"),
        
    }
}