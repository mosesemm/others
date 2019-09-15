
const express  = require('express');
const app = express();
const env = require('./util/env');

const permissionsRoutes = require("./routes/permissions-routes");

app.use(env.APP_CONTEXT, express.static("public"));
app.use(env.APP_CONTEXT, express.static("dist"));

app.use(env.APP_CONTEXT+"/api/permissions", permissionsRoutes);

app.listen(env.APP_PORT, () => {
   console.log(`App started listening on port ${env.APP_PORT}`) ;
});