<!DOCTYPE html>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %> 
<%@ taglib prefix="s" uri="/struts-tags" %>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Started with Struts</title>
        <s:head />
    </head>
<body>
<h2>Welcome to learning Struts</h2>
<s:url action="hello" var="helloUrl" >
    <s:param name="userName" > John Doe </s:param>
</s:url>
<p>
    <a href="<s:url action='hello' />" > Hello World</a>
    <a href="${helloUrl}">Second url</a>
</p>
<p>Another greeting using form.</p>
<s:form action="hello">
    <s:textfield name="userName" label="Your name" />
    <s:submit value="Submit" />
</s:form>
<p>
    <s:url action="registerInput" var="registerUrl" />
    <s:a href="%{registerUrl}">Please register</s:a> for amazing prices.
</p>
<p>
    <s:url action="registerInput" var="registerUrlSepedi">
        <s:param name="request_locale">nso</s:param>
    </s:url>
    <s:a href="%{registerUrlSepedi}">Ngwadisa ka Sepedi sa geno.</s:a> gore o hwetse di mpho.
</p>
<p>
    <a href="<s:url action='index' namespace='config-browser' />">Launch the configuration browser</a>
</p>
<p>
    <s:url action="index" var="indexUrl">
        <s:param name="debug">browser</s:param>
    </s:url>
    <a href="${indexUrl}">Reload this page with debugging</a>
</p>

<hr/>
<s:text name="contact" />
<s:debug/>
</body>
</html>
