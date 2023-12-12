<!DOCTYPE html>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ taglib prefix="s" uri="/struts-tags" %>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Hello World...</title>
    </head>
    <body>
        <h1><s:text name="greeting" /></h1>
        <h2><s:property value="messageStore.message" /> </h2>
        <h5><s:property value="messageStore"/></h5>
        <div>
            <h4>Count will be: <s:property value="#session.helloCount" /></h4>
        </div>

        <p>
            <a href="<s:url action='index'/>">Home</a>
        </p>

        <hr/>
        <s:text name="contact" />
    </body>
</html>